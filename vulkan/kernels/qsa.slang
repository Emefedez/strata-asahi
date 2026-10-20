// vulkan/kernels/qsa.slang - metal/kernels/qsa.metal in Slang: QSA (sparse full attention) decode for a window of
// tokens, between the projections.  Same contract and layouts as the MSL (see there):
//   qsa_prep    q/k/indexer norms + RoPE, int8 KV append (fp16 scale per 64), the indexer keys (last threadgroup)
//   qsa_score   per token, block b <= n_bid: sum over the 4 indexer heads of relu(qidx . key_b)
//   qsa_topk    per token: min(n_kv, 2051) cells of largest score, ties to the lowest cell, ascending (radix select)
//   qsa_attn    split-K partials (chunk of 64 cells, kv head, token)
//   qsa_merge   log-sum-exp merge * sigmoid(gate), written as prepared activations
// The int8 K rows are read four values to a 4-byte load (Honeykrisp: narrow loads reach a quarter of the bandwidth).

#define QH 24
#define QKV 2
#define QD 256
#define QROT 64
#define IH 4
#define ID 128
#define QBUDGET 2051
#define QCH 64

[vk::constant_id(3)] const int QSA_NTOK = 1;
[vk::constant_id(24)] const bool qsa_mrope = false;

struct QsaArgs {
    uint pos0;
    uint pos_base;
    uint max_cells;
    uint n_tok;
    float eps;
};

// the four signed bytes of a word, low byte first
float4 sbytes(uint w) {
    return float4(float(int(w << 24) >> 24), float(int(w << 16) >> 24), float(int(w << 8) >> 24), float(int(w) >> 24));
}
uint okey(float f) {
    const uint u = asuint(f);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

groupshared float q_part[8];
groupshared float q_row[256];
groupshared float q_amax[8];

float tg_sum256(float v, int sg, int lane, int nsg) {
    v = WaveActiveSum(v);
    if (lane == 0) q_part[sg] = v;
    GroupMemoryBarrierWithGroupSync();
    float t = 0.0f;
    for (int i = 0; i < nsg; ++i) t += q_part[i];
    GroupMemoryBarrierWithGroupSync();
    return t;
}

// one row (width w <= 256 = threads): out = rope((rs * x) * g)  (RoPE on the first 64 dims, NEOX pairs i, i+32)
float norm_rope(float x, float* g, int tid, int sg, int lane, int w, float eps, float2* cs, uint absp, bool rope, int* mtab) {
    const float ss = tg_sum256(tid < w ? x * x : 0.0f, sg, lane, 8);
    const float rs = rsqrt(ss / float(w) + eps);
    const float y = tid < w ? (rs * x) * g[tid] : 0.0f;
    q_row[tid] = y;
    GroupMemoryBarrierWithGroupSync();
    float o = y;
    if (rope && tid < QROT) {
        const int p = tid & 31;
        const uint pp = qsa_mrope ? uint(mtab[absp * 3 + uint(p % 3)]) : absp;
        const float2 c = cs[pp * 32 + p];
        const float a = q_row[p], b = q_row[p + 32];
        o = tid < 32 ? a * c.x - b * c.y : a * c.y + b * c.x;
    }
    GroupMemoryBarrierWithGroupSync();
    return o;
}

struct PrepPush {
    float* qfull;
    float* kcur;
    float* vcur;
    float* iqraw;
    float* ikraw;
    float* qn;
    float* kn;
    float* iqn;
    float* ikn;
    float2* cs;
    float* qcur;
    float* qidx;
    int8_t* kq;
    int8_t* vq;
    half* ksc;
    half* vsc;
    float* tail;
    float* dead;
    float* pooled;
    QsaArgs a;
    int* mtab;
};

// grid (32 rows + 1, n_tok): rows 0..23 q heads, 24..25 k heads, 26..27 v heads, 28..31 indexer q heads;
// threadgroup 32 (x = 32, y = 0) is the indexer-key threadgroup.  256 threads.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_prep(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform PrepPush pc) {
    const QsaArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int r = int(tg.x);
    if (r == 32) {
        if (tg.y != 0) return;
        for (int t = 0; t < int(a.n_tok); ++t) {
            const uint cell = a.pos0 + uint(t);
            const float x = tid < ID ? pc.ikraw[t * ID + tid] : 0.0f;
            if (tid < ID) pc.tail[(cell & 3) * ID + tid] = x;
            AllMemoryBarrierWithGroupSync();
            if (cell == 0) {
                const float o = norm_rope(x, pc.ikn, tid, sg, lane, ID, a.eps, pc.cs, 0, false, pc.mtab);
                if (tid < ID) pc.dead[tid] = o;
            }
            if ((cell & 3) == 3) {
                float m = 0.0f;
                if (tid < ID) m = (((pc.tail[tid] + pc.tail[ID + tid]) + pc.tail[2 * ID + tid]) + pc.tail[3 * ID + tid]) * 0.25f;
                const uint b = cell >> 2;
                const float o = norm_rope(m, pc.ikn, tid, sg, lane, ID, a.eps, pc.cs, a.pos_base + 4 * b, true, pc.mtab);
                if (tid < ID) pc.pooled[b * ID + tid] = o;
            }
            AllMemoryBarrierWithGroupSync();
        }
        return;
    }
    const int t = int(tg.y);
    const uint cell = a.pos0 + uint(t), absp = a.pos_base + cell;
    if (r < QH) {
        const float x = pc.qfull[(t * QH + r) * 2 * QD + tid];
        const float o = norm_rope(x, pc.qn, tid, sg, lane, QD, a.eps, pc.cs, absp, true, pc.mtab);
        pc.qcur[(t * QH + r) * QD + tid] = o;
    } else if (r < QH + 2 * QKV) {
        const bool isv = r >= QH + QKV;
        const int h = r - QH - (isv ? QKV : 0);
        float x = (isv ? pc.vcur : pc.kcur)[(t * QKV + h) * QD + tid];
        if (!isv) x = norm_rope(x, pc.kn, tid, sg, lane, QD, a.eps, pc.cs, absp, true, pc.mtab);
        // int8 with an fp16 scale per 64 values: groups = simdgroup pairs (64 threads)
        float m = WaveActiveMax(abs(x));
        if (lane == 0) q_amax[sg] = m;
        GroupMemoryBarrierWithGroupSync();
        const int g = tid >> 6;
        m = max(q_amax[2 * g], q_amax[2 * g + 1]);
        const half sh = half(m / 127.0f);
        const float sf = float(sh);
        int q = 0;
        if (sf > 0.0f) q = clamp(int(rint(x / sf)), -127, 127);
        const uint rowi = cell * QKV + uint(h);
        (isv ? pc.vq : pc.kq)[rowi * QD + tid] = int8_t(q);
        if ((tid & 63) == 0) (isv ? pc.vsc : pc.ksc)[rowi * 4 + g] = sh;
    } else {
        const int h = r - QH - 2 * QKV;
        const float x = tid < ID ? pc.iqraw[(t * IH + h) * ID + tid] : 0.0f;
        const float o = norm_rope(x, pc.iqn, tid, sg, lane, ID, a.eps, pc.cs, absp, true, pc.mtab);
        if (tid < ID) pc.qidx[(t * IH + h) * ID + tid] = o;
    }
}

struct ScorePush {
    float* qidx;
    float* pooled;
    float* dead;
    float* scores;
    QsaArgs a;
};
// scores [n_tok][max_blocks]; grid (ceil(max_blocks / 8), n_tok), 256 threads: a simdgroup per block.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_score(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform ScorePush pc) {
    const QsaArgs a = pc.a;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int t = int(tg.y);
    const uint n_kv = a.pos0 + uint(t) + 1, n_bid = n_kv >> 2;
    const uint b = tg.x * 8 + uint(sg);
    if (b > n_bid) return;
    float* key = b == n_bid ? pc.dead : pc.pooled + b * ID;
    float k[4];
    [ForceUnroll] for (int i = 0; i < 4; ++i) k[i] = key[lane + 32 * i];
    float s = 0.0f;
    [ForceUnroll]
    for (int h = 0; h < IH; ++h) {
        float d = 0.0f;
        [ForceUnroll] for (int i = 0; i < 4; ++i) d = fma(pc.qidx[(t * IH + h) * ID + lane + 32 * i], k[i], d);
        d = WaveActiveSum(d);
        s += max(d, 0.0f);
    }
    if (b == n_bid && (n_kv & 3) != 0) s += 1e9f;
    if (lane == 0) pc.scores[uint(t) * (a.max_cells / 4 + 2) + b] = s;
}

groupshared uint q_hist[256];
groupshared uint q_sel[4];
groupshared uint q_pref[32];

struct TopkPush {
    float* scores;
    int* ids;
    QsaArgs a;
};

// the radix select shared by qsa_topk (digit found by one thread) and qsa_topk2 (parallel suffix scan): the
// threshold key T and how many cells with key == T to take
void radix_select(float* sc, uint n_kv, uint n_bid, uint nb, int tid, int sg, int lane, bool par, out uint T, out uint need_out) {
    uint prefix = 0, pmask = 0, need = QBUDGET;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) q_hist[tid] = 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint b = uint(tid); b < nb; b += 1024) {
            const uint k = okey(sc[b]);
            if ((k & pmask) == prefix) {
                const uint w = b < n_bid ? 4u : (n_kv & 3u);
                if (w != 0) InterlockedAdd(q_hist[(k >> shift) & 255u], w);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (!par) {
            if (tid == 0) {
                uint acc = 0, d = 0;
                for (int v = 255; v >= 0; --v) {
                    const uint h = q_hist[v];
                    if (acc + h >= need) { d = uint(v); break; }
                    acc += h;
                }
                q_sel[0] = d;
                q_sel[1] = need - acc;
            }
        } else {
            // thread i < 256 holds bin 255 - i: an exclusive prefix over i = the cells of the higher digits
            uint cnt = 0, ex = 0;
            if (tid < 256) {
                cnt = q_hist[255 - tid];
                ex = uint(WavePrefixSum(int(cnt)));
                if (lane == 31) q_pref[sg] = ex + cnt;
            }
            GroupMemoryBarrierWithGroupSync();
            if (tid < 256) {
                uint above = ex;
                for (int i = 0; i < sg; ++i) above += q_pref[i];
                if (above < need && above + cnt >= need) { q_sel[0] = 255u - uint(tid); q_sel[1] = need - above; }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        prefix |= q_sel[0] << shift;
        pmask |= 255u << shift;
        need = q_sel[1];
        GroupMemoryBarrierWithGroupSync();
    }
    T = prefix;
    need_out = need;
}

// ids [n_tok][QBUDGET], ascending.  One threadgroup of 1024 threads per token.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_topk(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform TopkPush pc) {
    const QsaArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int t = int(tg.x);
    const uint n_kv = a.pos0 + uint(t) + 1, n_bid = n_kv >> 2, nb = n_bid + 1;
    int* out = pc.ids + t * QBUDGET;
    float* sc = pc.scores + uint(t) * (a.max_cells / 4 + 2);
    if (n_kv <= QBUDGET) {
        for (uint c = uint(tid); c < n_kv; c += 1024) out[c] = int(c);
        return;
    }
    uint T, need;
    radix_select(sc, n_kv, n_bid, nb, tid, sg, lane, false, T, need);
    uint base = 0, eq_taken = 0;
    for (uint b0 = 0; b0 < nb; b0 += 1024) {
        const uint b = b0 + uint(tid);
        uint w = 0, weq = 0;
        if (b < nb) {
            const uint k = okey(sc[b]);
            const uint cells = b < n_bid ? 4u : (n_kv & 3u);
            if (k > T) w = cells;
            else if (k == T) weq = cells;
        }
        const uint eq_before = uint(WavePrefixSum(int(weq)));
        if (lane == 31) q_pref[sg] = eq_before + weq;
        GroupMemoryBarrierWithGroupSync();
        uint eq_sg = 0;
        for (int i = 0; i < sg; ++i) eq_sg += q_pref[i];
        uint eq_tot = 0;
        for (int i = 0; i < 32; ++i) eq_tot += q_pref[i];
        GroupMemoryBarrierWithGroupSync();
        const uint eq0 = eq_taken + eq_sg + eq_before;
        uint take_eq = 0;
        if (weq != 0 && eq0 < need) take_eq = min(weq, need - eq0);
        const uint mine = w + take_eq;
        const uint before = uint(WavePrefixSum(int(mine)));
        if (lane == 31) q_pref[sg] = before + mine;
        GroupMemoryBarrierWithGroupSync();
        uint off = base;
        for (int i = 0; i < sg; ++i) off += q_pref[i];
        uint tot = 0;
        for (int i = 0; i < 32; ++i) tot += q_pref[i];
        GroupMemoryBarrierWithGroupSync();
        off += before;
        for (uint c = 0; c < mine; ++c) out[off + c] = int(4 * b + c);
        base += tot;
        eq_taken += eq_tot;
    }
}

struct AttnPush {
    float* qcur;
    int8_t* kq;
    int8_t* vq;
    half* ksc;
    half* vsc;
    int* ids;
    float* part;
    QsaArgs a;
};
groupshared float q_qs[12][QD];
groupshared float q_s[12][QCH];
groupshared int q_cid[QCH];
groupshared float q_mh[12];
groupshared float q_lh[12];

// partials: [n_tok][chunks][QH][QD + 2] (acc, then m, l).  grid (chunks, QKV, n_tok), 256 threads.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_attn(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform AttnPush pc) {
    const int HPG = QH / QKV;
    const QsaArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH;
    const int nchunks = (QBUDGET + QCH - 1) / QCH;
    float* pp = pc.part + ((uint(t) * nchunks + chunk) * QH) * (QD + 2);
    if (uint(c0) >= width) {
        for (int h = 0; h < HPG; ++h)
            if (tid == 0) { pp[(g * HPG + h) * (QD + 2) + QD] = -INFINITY; pp[(g * HPG + h) * (QD + 2) + QD + 1] = 0.0f; }
        return;
    }
    const int nc = min(QCH, int(width) - c0);
    for (int h = 0; h < HPG; ++h) q_qs[h][tid] = pc.qcur[(t * QH + g * HPG + h) * QD + tid];
    if (tid < QCH) q_cid[tid] = tid < nc ? pc.ids[t * QBUDGET + c0 + tid] : 0;
    GroupMemoryBarrierWithGroupSync();
    for (int c = sg; c < nc; c += 8) {
        const uint rowi = uint(q_cid[c]) * QKV + uint(g);
        uint* kr = (uint*)(pc.kq + rowi * QD + lane * 8);
        const float scl = float(pc.ksc[rowi * 4 + (lane >> 3)]);
        const float4 k0 = sbytes(kr[0]) * scl, k1 = sbytes(kr[1]) * scl;
        const float kv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
        for (int h = 0; h < HPG; ++h) {
            float d = 0.0f;
            [ForceUnroll] for (int i = 0; i < 8; ++i) d = fma(q_qs[h][lane * 8 + i], kv[i], d);
            d = WaveActiveSum(d);
            if (lane == 0) q_s[h][c] = d * 0.0625f;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (int h = sg; h < HPG; h += 8) {
        float m = -INFINITY;
        for (int c = lane; c < nc; c += 32) m = max(m, q_s[h][c]);
        m = WaveActiveMax(m);
        float l = 0.0f;
        for (int c = lane; c < nc; c += 32) {
            const float p = exp(q_s[h][c] - m);
            q_s[h][c] = p;
            l += p;
        }
        l = WaveActiveSum(l);
        if (lane == 0) { q_mh[h] = m; q_lh[h] = l; }
    }
    GroupMemoryBarrierWithGroupSync();
    float acc[12];
    [ForceUnroll] for (int h = 0; h < HPG; ++h) acc[h] = 0.0f;
    for (int c = 0; c < nc; ++c) {
        const uint rowi = uint(q_cid[c]) * QKV + uint(g);
        const float v = float(pc.vq[rowi * QD + tid]) * float(pc.vsc[rowi * 4 + (tid >> 6)]);
        [ForceUnroll] for (int h = 0; h < HPG; ++h) acc[h] = fma(q_s[h][c], v, acc[h]);
    }
    [ForceUnroll]
    for (int h = 0; h < HPG; ++h) {
        float* o = pp + (g * HPG + h) * (QD + 2);
        o[tid] = acc[h];
        if (tid == 0) { o[QD] = q_mh[h]; o[QD + 1] = q_lh[h]; }
    }
}

struct MergePush {
    float* part;
    float* qfull;
    float* xp;
    QsaArgs a;
};
groupshared float q_msum[256];

// the merge of `nc` chunks of the partials (chunk stride QH * (QD + 2)), * sigmoid(gate), as prepared activations
void merge_body(MergePush pc, uint3 tg, int tid, int nchunks, int ch) {
    const QsaArgs a = pc.a;
    const int h = int(tg.x), t = int(tg.y);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int nc = int((width + uint(ch) - 1) / uint(ch));
    float* pp = pc.part + (uint(t) * uint(nchunks) * QH + uint(h)) * (QD + 2);
    float M = -INFINITY;
    for (int k = 0; k < nc; ++k) M = max(M, pp[k * QH * (QD + 2) + QD]);
    float L = 0.0f, o = 0.0f;
    for (int k = 0; k < nc; ++k) {
        float* q = pp + k * QH * (QD + 2);
        const float e = exp(q[QD] - M);
        L = fma(q[QD + 1], e, L);
        o = fma(q[tid], e, o);
    }
    o = o / L;
    const float gate = pc.qfull[(t * QH + h) * 2 * QD + QD + tid];
    const float y = o * (1.0f / (1.0f + exp(-gate)));
    const uint n = QH * QD, plane = a.n_tok * n, d = uint(h) * QD + uint(tid);
    const int j = tid & 15;
    float* raw = pc.xp + uint(t) * n + d;
    raw[0] = y;
    raw[plane] = y * k_qsa_q2s[j & 7];
    raw[2 * plane] = y * k_qsa_p4s[j & 3];
    raw[3 * plane] = (j & 1) != 0 ? y * 0.00390625f : y;
    q_msum[tid] = y;
    GroupMemoryBarrierWithGroupSync();
    if (j == 0) {
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += q_msum[tid + k]; s1 += q_msum[tid + k + 1]; }
        pc.xp[4 * plane + uint(t) * (n / 16) + d / 16] = s0 + s1;
    }
}
static const float k_qsa_q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
static const float k_qsa_p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};

// grid (QH, n_tok), 256 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_merge(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform MergePush pc) {
    merge_body(pc, tg, int(tid), (QBUDGET + QCH - 1) / QCH, QCH);
}
