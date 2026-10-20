// vulkan/kernels/gr.slang - metal/kernels/gr.metal in Slang: the hyper-connection (gated residual) read for a window
// of tokens, with the previous sublayer's write folded in (fused_gr.hpp's math: FP32 activations, BF16 weights).
//   gr_norm  R'[c] = R[c] + bo * 2 sigmoid(inj_prev[c] / hc)       (when apply; written back in place)
//            xn[c] = R'[c] * w_norm[c] * rsqrt(mean(R'[c]^2) + eps)
//   gr_down  lo[k] = silu((w_down[k] . xn) / hc)   k < hc_lr;   inject[c] = w_inject[c] . xn
//   gr_up    mixed[d] = mean_c xn[c][d] * sigmoid(w_up[c n + d] . lo), written as prepared activations

[vk::constant_id(1)] const int GR_NTOK = 1;

struct GrArgs {
    uint n_embd;
    uint hc;
    uint hc_lr;
    uint n_tok;
    uint apply;
    float eps;
};

groupshared float g_part[8];

struct GrNormPush {
    float* R;
    float* bo;
    float* inj;
    float* w_norm;
    float* xn;
    GrArgs a;
};
// one threadgroup per (stream c, token t), 256 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gr_norm(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrNormPush pc) {
    const GrArgs a = pc.a;
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    float* r = pc.R + (t * a.hc + c) * n;
    const float g = a.apply != 0 ? 2.0f / (1.0f + exp(-pc.inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        float v = r[d];
        if (a.apply != 0) {
            v = fma(pc.bo[t * n + d], g, v);
            r[d] = v;
        }
        ss = fma(v, v, ss);
    }
    ss = WaveActiveSum(ss);
    if (WaveGetLaneIndex() == 0) g_part[tid / 32] = ss;
    GroupMemoryBarrierWithGroupSync();
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += g_part[i];
    const float rs = rsqrt(tot / float(n) + a.eps);
    float* o = pc.xn + (t * a.hc + c) * n;
    float* w = pc.w_norm + c * n;
    for (uint d = tid; d < n; d += 256) o[d] = r[d] * w[d] * rs;
}

// one row's K-split partial over xn (units of format F)
void grd_acc<let F : int>(uint8_t* w, int nin, int ks, int KS, int lane, float* xn, inout float acc[8]) {
    const int UNIT = f_UNIT(F);
    const int U = nin / UNIT, u0 = ks * U / KS, u1 = (ks + 1) * U / KS;
    for (int u = u0 + lane; u < u1; u += 32) {
        float q[MAXU];
        float4 sc;
        Dec_run<F>(w, u, q, sc);
        for (int t = 0; t < GR_NTOK; ++t) {
            float xs[MAXU], xsum[2] = {0.0f, 0.0f};
            [unroll] for (int k = 0; k < f_UNIT(F); k += 16) load16(xn + t * nin + UNIT * u + k, xs, k);
            acc[t] += unit_combine(UNIT, q, sc, xs, xsum);
        }
    }
}

struct GrDownPush {
    uint8_t* w_down;
    uint8_t* w_inject;
    float* xn;
    float* lo;
    float* inject;
    GrArgs a;
};
groupshared float g_grred[16 * 8];

// rows: 0 .. hc_lr - 1 = w_down, hc_lr .. hc_lr + hc - 1 = w_inject.  SG simdgroups split the row's units KS ways;
// RS = SG / KS rows per threadgroup.  FI: the inject rows' format.
void gr_down_f<let SG : int, let KS : int, let F : int, let FI : int>(GrDownPush pc, uint3 tg, uint tid) {
    const int RS = SG / KS;
    const GrArgs a = pc.a;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int ks = sg / RS, rs = sg % RS;
    const int row = int(tg.x) * RS + rs;
    const int nrows = int(a.hc_lr + a.hc);
    const int rr = min(row, nrows - 1);
    const int nin = int(a.hc * a.n_embd);
    const uint rb = uint(nin / f_QK(F) * f_BB(F)), rbi = uint(nin / f_QK(FI) * f_BB(FI));
    float acc[8];
    [ForceUnroll] for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    if (rr < int(a.hc_lr)) grd_acc<F>(pc.w_down + uint64_t(rr) * rb, nin, ks, KS, lane, pc.xn, acc);
    else grd_acc<FI>(pc.w_inject + uint64_t(rr - int(a.hc_lr)) * rbi, nin, ks, KS, lane, pc.xn, acc);
    for (int t = 0; t < GR_NTOK; ++t) {
        const float v = WaveActiveSum(acc[t]);
        if (lane == 0) g_grred[(ks * RS + rs) * 8 + t] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    if (ks == 0 && lane < GR_NTOK && row < nrows) {
        const int t = lane;
        float v = 0.0f;
        for (int k = 0; k < KS; ++k) v += g_grred[(k * RS + rs) * 8 + t];
        if (row < int(a.hc_lr)) {
            const float z = v / float(a.hc);
            pc.lo[t * int(a.hc_lr) + row] = z / (1.0f + exp(-z));
        } else {
            pc.inject[t * int(a.hc) + row - int(a.hc_lr)] = v;
        }
    }
}

struct GrUpPush {
    uint8_t* w_up;
    float* lo;
    float* xn;
    float* xp;
    GrArgs a;
};
groupshared float g_gate[8][4][16];
groupshared float g_msum[8][16];

// A threadgroup owns 16 consecutive d for all hc = 4 streams: 8 simdgroups, simdgroup s -> stream s / 2 and
// d0 + 8 (s % 2) .. + 8; LPR = 4 lanes per row.  XP: [4 kinds][n_tok][n] then S16 [n_tok][n / 16].
void gr_up_f<let F : int>(GrUpPush pc, uint3 tg, uint tid) {
    const GrArgs a = pc.a;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int n = int(a.n_embd), c = sg >> 1, dl = 8 * (sg & 1) + lane / 4, li = lane & 3;
    const int d = int(tg.x) * 16 + dl;
    const int UNIT = f_UNIT(F);
    uint8_t* w = pc.w_up + uint64_t(c * n + d) * (a.hc_lr / f_QK(F) * f_BB(F));
    const int U = int(a.hc_lr) / UNIT;
    float acc[8];
    [ForceUnroll] for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int u = li; u < U; u += 4) {
        float q[MAXU];
        float4 sc;
        Dec_run<F>(w, u, q, sc);
        for (int t = 0; t < GR_NTOK; ++t) {
            float xs[MAXU], xsum[2] = {0.0f, 0.0f};
            [unroll] for (int k = 0; k < f_UNIT(F); k += 16) load16(pc.lo + t * int(a.hc_lr) + UNIT * u + k, xs, k);
            acc[t] += unit_combine(UNIT, q, sc, xs, xsum);
        }
    }
    for (int t = 0; t < GR_NTOK; ++t) {
        const float v = lane_sum<4>(acc[t]);
        if (li == 0) g_gate[t][c][dl] = 1.0f / (1.0f + exp(-v));
    }
    GroupMemoryBarrierWithGroupSync();
    // 16 threads per token: thread j -> d0 + j
    const int t = int(tid) / 16, j = int(tid) & 15;
    if (t < GR_NTOK) {
        const int dd = int(tg.x) * 16 + j;
        const int hc = int(a.hc);
        float m = 0.0f;
        for (int cc = 0; cc < hc; ++cc) m = fma(pc.xn[(t * hc + cc) * n + dd], g_gate[t][cc][j], m);
        m = m / float(hc);
        const uint plane = a.n_tok * uint(n);
        float* raw = pc.xp + t * n + dd;
        raw[0] = m;
        raw[plane] = m * k_gr_q2s[j & 7];
        raw[2 * plane] = m * k_gr_p4s[j & 3];
        raw[3 * plane] = (j & 1) != 0 ? m * 0.00390625f : m;
        g_msum[t][j] = m;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < GR_NTOK && j == 0) {
        float s0 = 0.0f, s1 = 0.0f;   // the per-16 sum in xprep's order (even and odd partial sums, then added)
        for (int k = 0; k < 16; k += 2) { s0 += g_msum[t][k]; s1 += g_msum[t][k + 1]; }
        pc.xp[4 * a.n_tok * uint(n) + t * (n / 16) + tg.x] = s0 + s1;
    }
}
static const float k_gr_q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
static const float k_gr_p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};

#define GRD1(N, S, K, F, FI)                                                                                          \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void N##_##S##_##K(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrDownPush pc) {                     \
        gr_down_f<S, K, F, FI>(pc, tg, tid);                                                                         \
    }
#define GRD(S, K) GRD1(gr_down, S, K, 30, 30) GRD1(gr_down_q8, S, K, 8, 8) GRD1(gr_down_q8f, S, K, 8, 0)
GRD(8, 8) GRD(8, 4) GRD(4, 4) GRD(4, 2) GRD(16, 16) GRD(16, 8)
[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void gr_up(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrUpPush pc) { gr_up_f<30>(pc, tg, tid); }
[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void gr_up_q8(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrUpPush pc) { gr_up_f<8>(pc, tg, tid); }
