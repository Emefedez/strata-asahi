// vulkan/kernels/qsa2.slang - metal/kernels/qsa2.metal in Slang (after qsa.slang): QSA decode v2, more parallel.
//   qsa_attn2   chunks of 32 cells, the 12 query heads of a kv head in 2 groups of 6; V loads 8 cells at a time
//   qsa_merge2  qsa_merge for that chunking
//   qsa_topk2   the radix digit by a parallel suffix scan; cells emitted from contiguous block ranges per thread
//   qsa_attn3   qsa_attn's chunking without a reduction per (cell, head)

#define QCH2 32
#define QHG 2

groupshared float q2_qs[6][QD];
groupshared float q2_s[6][QCH2];
groupshared float q2_mh[6];
groupshared float q2_lh[6];
groupshared int q2_cid[QCH2];

// partials: [n_tok][chunks2][QH][QD + 2].  grid (chunks2, QKV * QHG, n_tok), 256 threads.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_attn2(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform AttnPush pc) {
    const int HPG = QH / QKV / QHG;
    const QsaArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int chunk = int(tg.x), g = int(tg.y) / QHG, hg = int(tg.y) % QHG, t = int(tg.z);
    const int h0 = g * (QH / QKV) + hg * HPG;
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH2;
    const int nchunks = (QBUDGET + QCH2 - 1) / QCH2;
    if (uint(c0) >= width) return;
    float* pp = pc.part + ((uint(t) * nchunks + chunk) * QH) * (QD + 2);
    const int nc = min(QCH2, int(width) - c0);
    for (int h = 0; h < HPG; ++h) q2_qs[h][tid] = pc.qcur[(t * QH + h0 + h) * QD + tid];
    if (tid < QCH2) q2_cid[tid] = tid < nc ? pc.ids[t * QBUDGET + c0 + tid] : 0;
    GroupMemoryBarrierWithGroupSync();
    for (int c = sg; c < nc; c += 8) {
        const uint rowi = uint(q2_cid[c]) * QKV + uint(g);
        uint* kr = (uint*)(pc.kq + rowi * QD + lane * 8);
        const float scl = float(pc.ksc[rowi * 4 + (lane >> 3)]);
        const float4 k0 = sbytes(kr[0]), k1 = sbytes(kr[1]);
        const float kv[8] = {k0.x * scl, k0.y * scl, k0.z * scl, k0.w * scl, k1.x * scl, k1.y * scl, k1.z * scl, k1.w * scl};
        [ForceUnroll]
        for (int h = 0; h < HPG; ++h) {
            float d = 0.0f;
            [ForceUnroll] for (int i = 0; i < 8; ++i) d = fma(q2_qs[h][lane * 8 + i], kv[i], d);
            d = WaveActiveSum(d);
            if (lane == 0) q2_s[h][c] = d * 0.0625f;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (sg < HPG) {
        const int h = sg;
        const float sv = lane < nc ? q2_s[h][lane] : -INFINITY;
        const float m = WaveActiveMax(sv);
        const float p = lane < nc ? exp(sv - m) : 0.0f;
        const float l = WaveActiveSum(p);
        if (lane < nc) q2_s[h][lane] = p;
        if (lane == 0) { q2_mh[h] = m; q2_lh[h] = l; }
    }
    GroupMemoryBarrierWithGroupSync();
    float acc[6];
    [ForceUnroll] for (int h = 0; h < HPG; ++h) acc[h] = 0.0f;
    for (int cb = 0; cb < nc; cb += 8) {
        float v[8];
        const int m = min(8, nc - cb);
        [ForceUnroll]
        for (int c = 0; c < 8; ++c) {
            v[c] = 0.0f;
            if (c < m) {
                const uint rowi = uint(q2_cid[cb + c]) * QKV + uint(g);
                v[c] = float(pc.vq[rowi * QD + tid]) * float(pc.vsc[rowi * 4 + (tid >> 6)]);
            }
        }
        for (int c = 0; c < m; ++c)
            [ForceUnroll] for (int h = 0; h < HPG; ++h) acc[h] = fma(q2_s[h][cb + c], v[c], acc[h]);
    }
    [ForceUnroll]
    for (int h = 0; h < HPG; ++h) {
        float* o = pp + (h0 + h) * (QD + 2);
        o[tid] = acc[h];
        if (tid == 0) { o[QD] = q2_mh[h]; o[QD + 1] = q2_lh[h]; }
    }
}

[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_merge2(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform MergePush pc) {
    merge_body(pc, tg, int(tid), (QBUDGET + QCH2 - 1) / QCH2, QCH2);
}

groupshared uint q2_sgs[32];
groupshared uint q2_sgs2[32];

[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_topk2(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform TopkPush pc) {
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
    radix_select(sc, n_kv, n_bid, nb, tid, sg, lane, true, T, need);
    const uint per = (nb + 1023) / 1024, b0 = min(nb, uint(tid) * per), b1 = min(nb, b0 + per);
    uint gt = 0, eq = 0;
    for (uint b = b0; b < b1; ++b) {
        const uint k = okey(sc[b]);
        const uint cells = b < n_bid ? 4u : (n_kv & 3u);
        if (k > T) gt += cells;
        else if (k == T) eq += cells;
    }
    const uint eqb = uint(WavePrefixSum(int(eq)));
    if (lane == 31) q2_sgs2[sg] = eqb + eq;
    GroupMemoryBarrierWithGroupSync();
    uint eq_before = eqb;
    for (int i = 0; i < sg; ++i) eq_before += q2_sgs2[i];
    const uint eq_take = eq_before >= need ? 0u : min(eq, need - eq_before);
    const uint mine = gt + eq_take;
    const uint mb = uint(WavePrefixSum(int(mine)));
    if (lane == 31) q2_sgs[sg] = mb + mine;
    GroupMemoryBarrierWithGroupSync();
    uint off = mb;
    for (int i = 0; i < sg; ++i) off += q2_sgs[i];
    uint left = eq_take;
    for (uint b = b0; b < b1; ++b) {
        const uint k = okey(sc[b]);
        const uint cells = b < n_bid ? 4u : (n_kv & 3u);
        uint take = 0;
        if (k > T) take = cells;
        else if (k == T) { take = min(cells, left); left -= take; }
        for (uint c = 0; c < take; ++c) out[off + c] = int(4 * b + c);
        off += take;
    }
}

// qsa_attn3: for q.k a lane owns a cell (8 simdgroups = 2 halves of the 64 cells x 4 groups of 3 heads), for p.v a
// thread owns a dimension.  grid (33, QKV, n_tok), 256 threads.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void qsa_attn3(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform AttnPush pc) {
    const int HPG = QH / QKV;
    const QsaArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH;
    const int nchunks = (QBUDGET + QCH - 1) / QCH;
    float* pp = pc.part + ((uint(t) * nchunks + chunk) * QH) * (QD + 2);
    if (uint(c0) >= width) return;
    const int nc = min(QCH, int(width) - c0);
    for (int h = 0; h < HPG; ++h) q_qs[h][tid] = pc.qcur[(t * QH + g * HPG + h) * QD + tid];
    if (tid < QCH) q_cid[tid] = tid < nc ? pc.ids[t * QBUDGET + c0 + tid] : 0;
    GroupMemoryBarrierWithGroupSync();
    {
        const int half_ = sg & 1, hgrp = sg >> 1;
        const int c = 32 * half_ + lane;
        if (c < nc) {
            const uint rowi = uint(q_cid[c]) * QKV + uint(g);
            uint* kr = (uint*)(pc.kq + rowi * QD);
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            for (int gq = 0; gq < 4; ++gq) {
                const float scl = float(pc.ksc[rowi * 4 + gq]);
                float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const float4 k4 = sbytes(kr[gq * 16 + i]);
                    const int base = gq * 64 + 4 * i;
                    const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
                    [ForceUnroll]
                    for (int e = 0; e < 4; ++e) {
                        e0 = fma(q_qs[3 * hgrp][base + e], kv[e], e0);
                        e1 = fma(q_qs[3 * hgrp + 1][base + e], kv[e], e1);
                        e2 = fma(q_qs[3 * hgrp + 2][base + e], kv[e], e2);
                    }
                }
                d0 = fma(e0, scl, d0);
                d1 = fma(e1, scl, d1);
                d2 = fma(e2, scl, d2);
            }
            q_s[3 * hgrp][c] = d0 * 0.0625f;
            q_s[3 * hgrp + 1][c] = d1 * 0.0625f;
            q_s[3 * hgrp + 2][c] = d2 * 0.0625f;
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
    for (int cb = 0; cb < nc; cb += 8) {
        float v[8];
        const int m = min(8, nc - cb);
        [ForceUnroll]
        for (int c = 0; c < 8; ++c) {
            v[c] = 0.0f;
            if (c < m) {
                const uint rowi = uint(q_cid[cb + c]) * QKV + uint(g);
                v[c] = float(pc.vq[rowi * QD + tid]) * float(pc.vsc[rowi * 4 + (tid >> 6)]);
            }
        }
        for (int c = 0; c < m; ++c)
            [ForceUnroll] for (int h = 0; h < HPG; ++h) acc[h] = fma(q_s[h][cb + c], v[c], acc[h]);
    }
    [ForceUnroll]
    for (int h = 0; h < HPG; ++h) {
        float* o = pp + (g * HPG + h) * (QD + 2);
        o[tid] = acc[h];
        if (tid == 0) { o[QD] = q_mh[h]; o[QD + 1] = q_lh[h]; }
    }
}
