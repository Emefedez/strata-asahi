// vulkan/kernels/qsa_prompt.slang - metal/kernels/qsa_prompt.metal in Slang (after qsa.slang): QSA attention for a
// PROMPT chunk.
//   qsa_q_f16     qcur (fp32) -> fp16
//   qsa_score_mm  qsa_score as a GEMM (fp32 operands): S[(t, h)][b] = qidx[t][h] . pooled[b] for 16 tokens x IH heads
//                 by 64 blocks a threadgroup, the epilogue summing relu over the heads; the tail block n_bid by one
//                 subgroup a token (the first column threadgroup); blocks past n_bid are not written
//   qsa_attn_mm   qsa_attn's split-K partials with fp16 operands (q, k = int8 * scale, p rounded to fp16)
//   qsa_flash     the whole attention of a (kv head, token) with an online softmax -> xq as fp16
// On the Mac qsa_attn_mm and qsa_flash run on the matrix units and are chosen with the M5's layout (coop_k4); the
// Vulkan engine takes the v1 path (qsa_attn + qsa_merge), so these two are plain SIMT kernels with the same operand
// roundings, for the parity test and as a reference.

[vk::constant_id(7)] const bool COOP_K4 = false;
[vk::constant_id(8)] const int QSA_STOP = 0;

struct QF16Push {
    float* q;
    half* qh;
    uint n;
};

[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_q_f16(uint gid : SV_DispatchThreadID, uniform QF16Push pc) {
    if (gid < pc.n) pc.qh[gid] = half(pc.q[gid]);
}

// ---- qsa_score_mm --------------------------------------------------------------------------------------------------
struct ScoreMmPush {
    float* qidx;
    float* pooled;
    float* dead;
    float* scores;
    QsaArgs a;
};

groupshared float g_sm_a[64 * 33];   // 64 rows (16 tokens x IH heads) x 32 k (+1: no bank conflicts)
groupshared float g_sm_b[64 * 33];   // 64 blocks x 32 k

// grid (ceil(blocks / 64), ceil(T / 16)), 128 threads; thread: block j = tid % 64, tokens tt = tid / 64 + 2 i
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_score_mm(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform ScoreMmPush pc) {
    const QsaArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int b0 = int(tg.x) * 64, t0 = int(tg.y) * 16;
    const int nb_all = int((a.pos0 + a.n_tok) / 4 + 1);
    const int rows = int(a.n_tok) * IH;
    const int j = tid % 64, tb = tid / 64;
    float acc[8][IH];
    [ForceUnroll] for (int i = 0; i < 8; ++i)
        [ForceUnroll] for (int h = 0; h < IH; ++h) acc[i][h] = 0.0f;
    for (int k0 = 0; k0 < ID; k0 += 32) {
        GroupMemoryBarrierWithGroupSync();
        for (int e = tid; e < 64 * 32; e += 128) {
            const int r = e / 32, c = e % 32;
            const int ra = t0 * IH + r, rb = b0 + r;
            g_sm_a[r * 33 + c] = ra < rows ? pc.qidx[ra * ID + k0 + c] : 0.0f;
            g_sm_b[r * 33 + c] = rb < nb_all ? pc.pooled[rb * ID + k0 + c] : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        for (int c = 0; c < 32; ++c) {
            const float bv = g_sm_b[j * 33 + c];
            [ForceUnroll] for (int i = 0; i < 8; ++i)
                [ForceUnroll] for (int h = 0; h < IH; ++h) acc[i][h] = fma(g_sm_a[((tb + 2 * i) * IH + h) * 33 + c], bv, acc[i][h]);
        }
    }
    const uint stride = a.max_cells / 4 + 2;
    [ForceUnroll]
    for (int i = 0; i < 8; ++i) {
        const int tt = tb + 2 * i, t = t0 + tt, b = b0 + j;
        if (t >= int(a.n_tok)) continue;
        const uint n_kv = a.pos0 + uint(t) + 1, n_bid = n_kv >> 2;
        if (uint(b) >= n_bid) continue;
        float s = 0.0f;
        [ForceUnroll] for (int h = 0; h < IH; ++h) s += max(acc[i][h], 0.0f);
        pc.scores[uint(t) * stride + uint(b)] = s;
    }
    // the tail block of each token (the first column threadgroup does it)
    if (tg.x == 0)
        for (int tt = sg; tt < 16; tt += 4) {
            const int t = t0 + tt;
            if (t >= int(a.n_tok)) break;
            const uint n_kv = a.pos0 + uint(t) + 1, n_bid = n_kv >> 2;
            float k[4];
            [ForceUnroll] for (int i = 0; i < 4; ++i) k[i] = pc.dead[lane + 32 * i];
            float s = 0.0f;
            for (int h = 0; h < IH; ++h) {
                float d = 0.0f;
                [ForceUnroll] for (int i = 0; i < 4; ++i) d = fma(pc.qidx[(t * IH + h) * ID + lane + 32 * i], k[i], d);
                d = WaveActiveSum(d);
                s += max(d, 0.0f);
            }
            if ((n_kv & 3) != 0) s += 1e9f;
            if (lane == 0) pc.scores[uint(t) * stride + n_bid] = s;
        }
}

// ---- the fp16-operand attention, SIMT ------------------------------------------------------------------------------
// k / v of cell `id` (kv head g), dimension d, as the matrix units see them: int8 * fp16 scale, rounded to fp16
float qp_k(int8_t* kq, half* ksc, int id, int g, int d) {
    const uint row = uint(id) * QKV + uint(g);
    return float(half(float(kq[row * QD + uint(d)]) * float(ksc[row * 4 + uint(d >> 6)])));
}

struct AttnMmPush {
    half* qh;
    int8_t* kq;
    int8_t* vq;
    half* ksc;
    half* vsc;
    int* ids;
    float* part;
    QsaArgs a;
};

groupshared int g_am_cid[64];
groupshared float g_am_s[12 * 64];
groupshared float g_am_p[12 * 64];

// grid (chunks, QKV, n_tok), 32 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_attn_mm(uint3 tg : SV_GroupID, uint lane : SV_GroupIndex, uniform AttnMmPush pc) {
    const int HPG = QH / QKV, NCH = (QBUDGET + QCH - 1) / QCH;
    const QsaArgs a = pc.a;
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH;
    float* pp = pc.part + ((uint(t) * uint(NCH) + uint(chunk)) * QH + uint(g * HPG)) * (QD + 2);
    if (uint(c0) >= width) {
        if (lane < uint(HPG)) { pp[lane * (QD + 2) + QD] = -INFINITY; pp[lane * (QD + 2) + QD + 1] = 0.0f; }
        return;
    }
    const int nc = min(QCH, int(width) - c0);
    for (int c = int(lane); c < QCH; c += 32) g_am_cid[c] = c < nc ? pc.ids[t * QBUDGET + c0 + c] : -1;
    GroupMemoryBarrierWithGroupSync();
    if (QSA_STOP == 1) { if (lane < uint(HPG)) { pp[lane * (QD + 2) + QD] = 1.0f; pp[lane * (QD + 2) + QD + 1] = 1.0f; } return; }
    // S: lane = cell (two cells a lane), the 12 heads' dots over the 256 dims
    half* q = pc.qh + (uint(t) * QH + uint(g * HPG)) * QD;
    for (int c = int(lane); c < QCH; c += 32) {
        const int id = g_am_cid[c];
        float s[12];
        [ForceUnroll] for (int h = 0; h < 12; ++h) s[h] = 0.0f;
        if (id >= 0)
            for (int d = 0; d < QD; ++d) {
                const float kv = qp_k(pc.kq, pc.ksc, id, g, d);
                [ForceUnroll] for (int h = 0; h < 12; ++h) s[h] = fma(float(q[h * QD + d]), kv, s[h]);
            }
        [ForceUnroll] for (int h = 0; h < 12; ++h) g_am_s[h * QCH + c] = s[h];
    }
    GroupMemoryBarrierWithGroupSync();
    if (QSA_STOP == 2) { if (lane < uint(HPG)) { pp[lane * (QD + 2) + QD] = 2.0f; pp[lane * (QD + 2) + QD + 1] = 1.0f; } return; }
    float mh = -INFINITY, lh = 0.0f;
    if (lane < uint(HPG)) {
        for (int c = 0; c < nc; ++c) mh = max(mh, g_am_s[lane * QCH + c] * 0.0625f);
        for (int c = 0; c < QCH; ++c) {
            const float e = c < nc ? exp(g_am_s[lane * QCH + c] * 0.0625f - mh) : 0.0f;
            lh += e;
            g_am_p[lane * QCH + c] = float(half(e));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (QSA_STOP == 3) { if (lane < uint(HPG)) { pp[lane * (QD + 2) + QD] = 3.0f; pp[lane * (QD + 2) + QD + 1] = 1.0f; } return; }
    // acc = P V: lane = dimension (8 a lane)
    for (int i = 0; i < QD / 32; ++i) {
        const int d = int(lane) + 32 * i;
        float acc[12];
        [ForceUnroll] for (int h = 0; h < 12; ++h) acc[h] = 0.0f;
        for (int c = 0; c < nc; ++c) {
            const int id = g_am_cid[c];
            const uint row = uint(id) * QKV + uint(g);
            const float vv = float(half(float(pc.vq[row * QD + uint(d)]) * float(pc.vsc[row * 4 + uint(d >> 6)])));
            [ForceUnroll] for (int h = 0; h < 12; ++h) acc[h] = fma(g_am_p[h * QCH + c], vv, acc[h]);
        }
        [ForceUnroll] for (int h = 0; h < 12; ++h) pp[h * (QD + 2) + d] = acc[h];
    }
    if (lane < uint(HPG)) { pp[lane * (QD + 2) + QD] = mh; pp[lane * (QD + 2) + QD + 1] = lh; }
}

struct FlashPush {
    half* qh;
    int8_t* kq;
    int8_t* vq;
    half* ksc;
    half* vsc;
    int* ids;
    float* qfull;
    half* xqh;
    QsaArgs a;
};

groupshared int g_fl_cid[64];
groupshared float g_fl_s[12 * 64];
groupshared float g_fl_p[12 * 64];
groupshared float g_fl_fac[12];
groupshared float g_fl_l[12];
groupshared float g_fl_m[12];

// grid (QKV, n_tok), 128 threads; thread: dimensions tid and tid + 128 of the 12 heads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_flash(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform FlashPush pc) {
    const int HPG = QH / QKV;
    const QsaArgs a = pc.a;
    const int tid = int(tidu);
    const int g = int(tg.x), t = int(tg.y);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int nch = int((width + QCH - 1) / QCH);
    if (tid < 12) { g_fl_m[tid] = -INFINITY; g_fl_l[tid] = 0.0f; }
    half* q = pc.qh + (uint(t) * QH + uint(g * HPG)) * QD;
    float tot[2][12];
    [ForceUnroll] for (int i = 0; i < 2; ++i)
        [ForceUnroll] for (int h = 0; h < 12; ++h) tot[i][h] = 0.0f;
    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * QCH, nc = min(QCH, int(width) - c0);
        GroupMemoryBarrierWithGroupSync();
        if (tid < QCH) g_fl_cid[tid] = tid < nc ? pc.ids[t * QBUDGET + c0 + tid] : -1;
        GroupMemoryBarrierWithGroupSync();
        // S: 12 x 64 entries, 6 a thread (cell tid % 64, heads tid / 64 + 2 r)
        {
            const int c = tid % 64, hb = tid / 64, id = g_fl_cid[c];
            float s[6];
            [ForceUnroll] for (int r = 0; r < 6; ++r) s[r] = 0.0f;
            if (id >= 0)
                for (int d = 0; d < QD; ++d) {
                    const float kv = qp_k(pc.kq, pc.ksc, id, g, d);
                    [ForceUnroll] for (int r = 0; r < 6; ++r) s[r] = fma(float(q[(hb + 2 * r) * QD + d]), kv, s[r]);
                }
            [ForceUnroll] for (int r = 0; r < 6; ++r) g_fl_s[(hb + 2 * r) * QCH + c] = s[r];
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < HPG) {
            float m = g_fl_m[tid];
            for (int c = 0; c < nc; ++c) m = max(m, g_fl_s[tid * QCH + c] * 0.0625f);
            const float f = exp(g_fl_m[tid] - m);
            float l = g_fl_l[tid] * f;
            for (int c = 0; c < QCH; ++c) {
                const float e = c < nc ? exp(g_fl_s[tid * QCH + c] * 0.0625f - m) : 0.0f;
                l += e;
                g_fl_p[tid * QCH + c] = float(half(e));
            }
            g_fl_m[tid] = m;
            g_fl_l[tid] = l;
            g_fl_fac[tid] = f;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (int i = 0; i < 2; ++i) {
            const int d = tid + 128 * i;
            float acc[12];
            [ForceUnroll] for (int h = 0; h < 12; ++h) acc[h] = 0.0f;
            for (int c = 0; c < nc; ++c) {
                const uint row = uint(g_fl_cid[c]) * QKV + uint(g);
                const float vv = float(half(float(pc.vq[row * QD + uint(d)]) * float(pc.vsc[row * 4 + uint(d >> 6)])));
                [ForceUnroll] for (int h = 0; h < 12; ++h) acc[h] = fma(g_fl_p[h * QCH + c], vv, acc[h]);
            }
            [ForceUnroll] for (int h = 0; h < 12; ++h) tot[i][h] = fma(tot[i][h], g_fl_fac[h], acc[h]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (int i = 0; i < 2; ++i) {
        const int d = tid + 128 * i;
        [ForceUnroll]
        for (int h = 0; h < 12; ++h) {
            const int hh = g * HPG + h;
            const float gate = pc.qfull[(uint(t) * QH + uint(hh)) * 2 * QD + QD + uint(d)];
            pc.xqh[(uint(t) * QH + uint(hh)) * QD + uint(d)] = half(tot[i][h] / g_fl_l[h] * (1.0f / (1.0f + exp(-gate))));
        }
    }
}
