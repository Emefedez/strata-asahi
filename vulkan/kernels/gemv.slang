// vulkan/kernels/gemv.slang - the dense GEMV of metal/kernels/gemv.metal in Slang: y[t][r] = W[r] . x[t] for a
// window of 1..8 tokens (dense projections, the shared expert, the LM head), any format of quant/units.
//
// Shape of a threadgroup: SG simdgroups; KS of them split the row's units (KS = 1: no split), the other SG/KS cover
// different rows. In a simdgroup, LPR lanes share a row and 32/LPR rows run side by side; each lane holds NR rows.
// The weights of a unit are decoded once and applied to every token. Partial sums of a K split are added in a fixed
// order (deterministic). Same entry-point names (gemv_<F>_<LPR>_<NR>_<SG>_<KS>) and function constants as the MSL.

struct GemvArgs {
    uint n_in;
    uint n_out;
    uint n_tok;
    uint row_bytes;
    uint x_stride;   // floats between tokens' x rows
    uint y_stride;   // floats between tokens' y rows
};
struct GemvPush {
    uint8_t* W;
    float* xp;
    float* s16;
    float* y;
    GemvArgs a;
};

[vk::constant_id(0)] const int NTOK = 1;         // tokens of the window (1..8): the token loops are specialised
// a bench probe (function constant 9): 1 = tokens past the first reuse token 0's x (no load; wrong values),
// 2 = tokens past the first skip the combine, 4 = no x loads, 5 = no weight loads or decode (all wrong values)
[vk::constant_id(9)] const int GEMV_PROBE = 0;

// function constant 10 (GEMV_XS): the window's x read through unit_dot (a float4 at a time, bitwise the same sums)
// instead of X_load's 32-value slice a token (default; 0: the slices, as the MSL)
[vk::constant_id(10)] const bool GEMV_XS = true;

groupshared float g_red[256];

void gemv<let F : int, let LPR : int, let NR : int, let SG : int, let KS : int>(GemvPush pc, uint3 tg, uint tid) {
    const int UNIT = f_UNIT(F), RS = SG / KS, RG = 32 / LPR;
    const GemvArgs a = pc.a;
    load_tab<F>(tid, SG * 32);
    GroupMemoryBarrierWithGroupSync();
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int ks = sg / RS, rs = sg % RS;
    const int li = lane % LPR;
    const int lrow0 = (rs * RG + lane / LPR) * NR;
    const int row0 = int(tg.x) * RS * RG * NR + lrow0;
    const int U = int(a.n_in) / UNIT;
    const int u0 = ks * U / KS, u1 = (ks + 1) * U / KS;
    float acc[NR][8];
    [ForceUnroll]
    for (int r = 0; r < NR; ++r)
        [ForceUnroll]
        for (int t = 0; t < 8; ++t) acc[r][t] = 0.0f;
    uint8_t* rp[NR];
    [ForceUnroll]
    for (int r = 0; r < NR; ++r) rp[r] = pc.W + uint64_t(min(row0 + r, int(a.n_out) - 1)) * a.row_bytes;
    const int s16n = int(a.n_in) / 16;
    if (GEMV_XS && GEMV_PROBE == 0) {
        for (int u = u0 + li; u < u1; u += LPR) {
            float q[NR][MAXU];
            float4 sc[NR];
            [ForceUnroll] for (int r = 0; r < NR; ++r) Dec_run<F>(rp[r], u, q[r], sc[r]);
            for (int t = 0; t < NTOK; ++t)
                [ForceUnroll] for (int r = 0; r < NR; ++r)
                    acc[r][t] += unit_dot<F>(pc.xp + t * int(a.x_stride), pc.s16 + t * s16n, u, q[r], sc[r]);
        }
    } else if (GEMV_PROBE == 3) {
        // rows outer, tokens inner: one row's decoded unit in registers at a time
        for (int u = u0 + li; u < u1; u += LPR)
            [ForceUnroll]
            for (int r = 0; r < NR; ++r) {
                float q[MAXU];
                float4 sc;
                Dec_run<F>(rp[r], u, q, sc);
                for (int t = 0; t < NTOK; ++t) {
                    float xs[MAXU], xsum[2];
                    X_load<F>(pc.xp + t * int(a.x_stride), pc.s16 + t * s16n, u, xs, xsum);
                    acc[r][t] += unit_combine(UNIT, q, sc, xs, xsum);
                }
            }
    } else
        for (int u = u0 + li; u < u1; u += LPR) {
            float q[NR][MAXU];
            float4 sc[NR];
            [ForceUnroll]
            for (int r = 0; r < NR; ++r) {
                if (GEMV_PROBE == 5) {   // bench probe: no weight loads or decode (wrong values)
                    [ForceUnroll] for (int j = 0; j < MAXU; ++j) q[r][j] = float(j + u);
                    sc[r] = float4(1, 1, 0, 0);
                } else
                    Dec_run<F>(rp[r], u, q[r], sc[r]);
            }
            float xs[MAXU], xsum[2];
            if (GEMV_PROBE == 4) {   // bench probe: no x loads (constant x; wrong values)
                [ForceUnroll] for (int j = 0; j < MAXU; ++j) xs[j] = float(j);
                xsum[0] = xsum[1] = 1.0f;
            }
            for (int t = 0; t < NTOK; ++t) {
                if ((GEMV_PROBE != 1 || t == 0) && GEMV_PROBE != 4) X_load<F>(pc.xp + t * int(a.x_stride), pc.s16 + t * s16n, u, xs, xsum);
                if (GEMV_PROBE != 2 || t == 0)
                    [ForceUnroll]
                    for (int r = 0; r < NR; ++r) acc[r][t] += unit_combine(UNIT, q[r], sc[r], xs, xsum);
            }
        }
    const int RPT = RS * RG * NR;
    [ForceUnroll]
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            if (KS == 1) {
                if (li == 0 && row0 + r < int(a.n_out)) pc.y[t * int(a.y_stride) + row0 + r] = v;
            } else if (li == 0) {
                g_red[(ks * RPT + lrow0 + r) * 8 + t] = v;
            }
        }
    if (KS > 1) {
        GroupMemoryBarrierWithGroupSync();
        if (ks == 0 && li == 0)
            [ForceUnroll]
            for (int r = 0; r < NR; ++r)
                for (int t = 0; t < NTOK; ++t) {
                    float v = 0.0f;
                    [ForceUnroll]
                    for (int k = 0; k < KS; ++k) v += g_red[(k * RPT + lrow0 + r) * 8 + t];
                    if (row0 + r < int(a.n_out)) pc.y[t * int(a.y_stride) + row0 + r] = v;
                }
    }
}

#define GV(F, L, R, S, K)                                                                            \
    [shader("compute")]                                                                             \
    [numthreads(TG_X, TG_Y, TG_Z)]                                                                  \
    void gemv_##F##_##L##_##R##_##S##_##K(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GemvPush pc) { \
        gemv<F, L, R, S, K>(pc, tg, tid);                                                           \
    }
#define GV_SET(F) GV(F, 32, 1, 4, 1) GV(F, 32, 2, 4, 1) GV(F, 16, 1, 4, 1) GV(F, 16, 2, 4, 1) GV(F, 8, 1, 4, 1) \
                  GV(F, 8, 2, 4, 1) GV(F, 4, 2, 4, 1) GV(F, 4, 1, 4, 1) GV(F, 32, 1, 4, 4) GV(F, 32, 2, 4, 2) \
                  GV(F, 32, 1, 8, 8) GV(F, 16, 1, 8, 4) GV(F, 32, 1, 8, 2) GV(F, 8, 1, 8, 1) GV(F, 1, 1, 4, 1) \
                  GV(F, 1, 2, 4, 1) GV(F, 2, 1, 4, 1) GV(F, 2, 2, 4, 1) GV(F, 1, 1, 2, 1) GV(F, 4, 4, 4, 1)
// the numeric type codes (as the MSL host names): T_F32 0, T_BF16 30, T_Q8_0 8, T_Q4_0 2, T_Q5_0 6, T_IQ4_NL 20,
// T_IQ4_XS 23, T_Q2_0 42, T_IQ3_S 21, T_Q3_K 11, T_Q4_K 12, T_Q5_K 13, T_Q6_K 14
GV_SET(0) GV_SET(30) GV_SET(8) GV_SET(2) GV_SET(6) GV_SET(20) GV_SET(23) GV_SET(42) GV_SET(21) GV_SET(11) GV_SET(12)
GV_SET(13) GV_SET(14)
