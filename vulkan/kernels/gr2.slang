// vulkan/kernels/gr2.slang - metal/kernels/gr2.metal in Slang (after quant/units/gr): the hyper-connection read in TWO
// dispatches, for BF16 HC weights or a Q8_0 copy of them (STRATA_HC_Q8).
//   gr_down3  grid (row blocks, hc streams): a threadgroup owns stream c, forms R'_c = R_c + bo * 2 sigmoid(inj_prev_c
//             / hc), its rs_c, stages xn_c in threadgroup memory and every simdgroup dots its rows' stream-c quarter
//             with it: D[t][row][c].  Row block 0 also writes R'_c to R_out and rs_c.
//   gr_up3    lo[k] = silu(sum_c D[t][k][c] / hc), then as gr_up with xn = R' * w_norm * rs on the fly.
//   gr_down3b BF16 only: gr_down3 with the rows' weights loaded into registers before the norm phase.
//   gr_norm_comb  gr_norm with bo formed on the fly as the MoE combine.

#ifndef P_TOKENT
#define P_TOKENT 532   // moe's plan layout
#endif

struct GrArgs3 {
    uint n_embd;
    uint hc;
    uint hc_lr;
    uint n_tok;
    uint apply;
    float eps;
    uint rb_down;
    uint rb_up;
    uint comb;
    uint topk;
};

float bo_at(float* bo, int* plan, float* ye, float* ysh, float* sg, GrArgs3 a, int t, int d) {
    const int n = int(a.n_embd);
    if (a.comb == 0) return bo[t * n + d];
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += ye[plan[P_TOKENT + t * int(a.topk) + j] * n + d];
    const float gs = 1.0f / (1.0f + exp(-sg[t]));
    return fma(gs, ysh[t * n + d], s);
}

// xn of TT tokens (TT = 1..3) in threadgroup memory: one array per TT, so a 1-token kernel holds 10 KB, not 30
groupshared float g_xs1[1 * 2560];
groupshared float g_xs2[2 * 2560];
groupshared float g_xs3[3 * 2560];
float xs_get<let TT : int>(int i) { return TT == 1 ? g_xs1[i] : (TT == 2 ? g_xs2[i] : g_xs3[i]); }
void xs_set<let TT : int>(int i, float v) {
    if (TT == 1) g_xs1[i] = v;
    else if (TT == 2) g_xs2[i] = v;
    else g_xs3[i] = v;
}
groupshared float g_p3[8][3];
groupshared float g_g3[3];
groupshared float g_rs3[3];

struct GrDown3Push {
    uint8_t* w_down;
    uint8_t* w_inject;
    float* Rin;
    float* Rout;
    float* bo;
    float* Dprev;
    float* wn;
    float* D;
    float* rs_out;
    GrArgs3 a;
    int* plan;
    float* ye;
    float* ysh;
    float* sgv;
};

// the norm phase of gr_down3 / gr_down3b for tokens t0 .. t0 + T - 1 of stream c: g, R' (and Rout), rs, xn
void down3_norm<let TT : int>(GrDown3Push pc, uint3 tg, int tid, int sg, int lane, int t0, int T) {
    const GrArgs3 a = pc.a;
    const int n = int(a.n_embd), hc = int(a.hc), NIN = hc * n, c = int(tg.y);
    const int nrows = int(a.hc_lr) + hc;
    if (tid < T) {
        float g = 0.0f;
        if (a.apply != 0) {
            float* dp = pc.Dprev + ((t0 + tid) * nrows + int(a.hc_lr) + c) * hc;
            const float ip = ((dp[0] + dp[1]) + dp[2]) + dp[3];
            g = 2.0f / (1.0f + exp(-ip / float(hc)));
        }
        g_g3[tid] = g;
    }
    GroupMemoryBarrierWithGroupSync();
    for (int t = 0; t < T; ++t) {
        float* rt = pc.Rin + (t0 + t) * NIN + c * n;
        float ss = 0.0f;
        for (int d = tid; d < n; d += 256) {
            float v = rt[d];
            if (a.apply != 0) v = fma(bo_at(pc.bo, pc.plan, pc.ye, pc.ysh, pc.sgv, a, t0 + t, d), g_g3[t], v);
            xs_set<TT>(t * 2560 + d, v);
            ss = fma(v, v, ss);
            if (tg.x == 0) pc.Rout[(t0 + t) * NIN + c * n + d] = v;
        }
        ss = WaveActiveSum(ss);
        if (lane == 0) g_p3[sg][t] = ss;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < T) {
        float tot = 0.0f;
        for (int k = 0; k < 8; ++k) tot += g_p3[k][tid];
        const float rsv = rsqrt(tot / float(n) + a.eps);
        g_rs3[tid] = rsv;
        if (tg.x == 0) pc.rs_out[(t0 + tid) * hc + c] = rsv;
    }
    GroupMemoryBarrierWithGroupSync();
    for (int t = 0; t < T; ++t)
        for (int d = tid; d < n; d += 256) xs_set<TT>(t * 2560 + d, xs_get<TT>(t * 2560 + d) * pc.wn[c * n + d] * g_rs3[t]);
    GroupMemoryBarrierWithGroupSync();
}

void gr_down3<let F : int, let TT : int, let RPS : int>(GrDown3Push pc, uint3 tg, uint tidu) {
    const int UNIT = f_UNIT(F);
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    load_tab<F>(tidu, 256);
    const GrArgs3 a = pc.a;
    const int n = int(a.n_embd), hc = int(a.hc), c = int(tg.y);
    const int nrows = int(a.hc_lr) + hc;
    for (int t0 = 0; t0 < GR_NTOK; t0 += TT) {
        const int T = min(TT, GR_NTOK - t0);
        down3_norm<TT>(pc, tg, tid, sg, lane, t0, T);
        int rows[RPS];
        uint8_t* wr[RPS];
        [ForceUnroll]
        for (int k = 0; k < RPS; ++k) {
            rows[k] = (int(tg.x) * 8 + sg) * RPS + k;
            const int rr = min(rows[k], nrows - 1);
            wr[k] = rr < int(a.hc_lr) ? pc.w_down + uint64_t(rr) * a.rb_down : pc.w_inject + uint64_t(rr - int(a.hc_lr)) * a.rb_down;
        }
        float acc[RPS][TT];
        [ForceUnroll] for (int k = 0; k < RPS; ++k)
            [ForceUnroll] for (int t = 0; t < TT; ++t) acc[k][t] = 0.0f;
        const int U = n / UNIT, u0 = c * U;
        for (int u = lane; u < U; u += 32) {
            float q[RPS][MAXU];
            float4 sc[RPS];
            [ForceUnroll] for (int k = 0; k < RPS; ++k) Dec_run<F>(wr[k], u0 + u, q[k], sc[k]);
            [ForceUnroll]
            for (int t = 0; t < TT; ++t) {
                if (t >= T) break;
                float xs[MAXU], xsum[2] = {0.0f, 0.0f};
                [unroll] for (int e = 0; e < f_UNIT(F); ++e) xs[e] = xs_get<TT>(t * 2560 + u * UNIT + e);
                [ForceUnroll] for (int k = 0; k < RPS; ++k) acc[k][t] += unit_combine(UNIT, q[k], sc[k], xs, xsum);
            }
        }
        [ForceUnroll]
        for (int k = 0; k < RPS; ++k)
            [ForceUnroll]
            for (int t = 0; t < TT; ++t) {
                if (t >= T) break;
                const float v = WaveActiveSum(acc[k][t]);
                if (lane == 0 && rows[k] < nrows) pc.D[((t0 + t) * nrows + rows[k]) * hc + c] = v;
            }
        GroupMemoryBarrierWithGroupSync();
    }
}

void gr_down3b<let TT : int, let RPS : int>(GrDown3Push pc, uint3 tg, uint tidu) {
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const GrArgs3 a = pc.a;
    const int n = int(a.n_embd), hc = int(a.hc), c = int(tg.y);
    const int nrows = int(a.hc_lr) + hc;
    int rows[RPS];
    uint4 wr[RPS][5][2];
    [ForceUnroll]
    for (int k = 0; k < RPS; ++k) {
        rows[k] = (int(tg.x) * 8 + sg) * RPS + k;
        const int rr = min(rows[k], nrows - 1);
        uint8_t* w = rr < int(a.hc_lr) ? pc.w_down + uint64_t(rr) * a.rb_down : pc.w_inject + uint64_t(rr - int(a.hc_lr)) * a.rb_down;
        uint4* p = (uint4*)(w + 2 * (c * n));
        [ForceUnroll]
        for (int i = 0; i < 5; ++i) {
            wr[k][i][0] = loadAligned<16>(p + 2 * (lane + 32 * i));
            wr[k][i][1] = loadAligned<16>(p + 2 * (lane + 32 * i) + 1);
        }
    }
    for (int t0 = 0; t0 < GR_NTOK; t0 += TT) {
        const int T = min(TT, GR_NTOK - t0);
        down3_norm<TT>(pc, tg, tid, sg, lane, t0, T);
        float acc[RPS][TT];
        [ForceUnroll] for (int k = 0; k < RPS; ++k)
            [ForceUnroll] for (int t = 0; t < TT; ++t) acc[k][t] = 0.0f;
        [ForceUnroll]
        for (int i = 0; i < 5; ++i) {
            const int u = lane + 32 * i;
            [ForceUnroll]
            for (int k = 0; k < RPS; ++k) {
                const uint v[8] = {wr[k][i][0].x, wr[k][i][0].y, wr[k][i][0].z, wr[k][i][0].w,
                                   wr[k][i][1].x, wr[k][i][1].y, wr[k][i][1].z, wr[k][i][1].w};
                float q[MAXU];
                [ForceUnroll]
                for (int e = 0; e < 8; ++e) {
                    q[2 * e] = asfloat(v[e] << 16);
                    q[2 * e + 1] = asfloat(v[e] & 0xFFFF0000u);
                }
                const float4 sc = float4(1.0f, 1.0f, 0.0f, 0.0f);
                [ForceUnroll]
                for (int t = 0; t < TT; ++t) {
                    if (t >= T) break;
                    float xs[MAXU], xsum[2] = {0.0f, 0.0f};
                    [ForceUnroll] for (int e = 0; e < 16; ++e) xs[e] = xs_get<TT>(t * 2560 + u * 16 + e);
                    acc[k][t] += unit_combine(16, q, sc, xs, xsum);
                }
            }
        }
        [ForceUnroll]
        for (int k = 0; k < RPS; ++k)
            [ForceUnroll]
            for (int t = 0; t < TT; ++t) {
                if (t >= T) break;
                const float v = WaveActiveSum(acc[k][t]);
                if (lane == 0 && rows[k] < nrows) pc.D[((t0 + t) * nrows + rows[k]) * hc + c] = v;
            }
        GroupMemoryBarrierWithGroupSync();
    }
}

struct GrUp3Push {
    uint8_t* w_up;
    float* D;
    float* Rn;
    float* xp;
    GrArgs3 a;
    float* wn;
    float* rs;
};
groupshared float g_lo3[8][320];

// grid (n_embd / 16), 256 threads.
void gr_up3<let F : int>(GrUp3Push pc, uint3 tg, uint tidu) {
    const int UNIT = f_UNIT(F);
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    load_tab<F>(tidu, 256);
    const GrArgs3 a = pc.a;
    const int n = int(a.n_embd), hc = int(a.hc), lr = int(a.hc_lr), nrows = lr + hc;
    for (int e = tid; e < GR_NTOK * lr; e += 256) {
        const int t = e / lr, k = e - t * lr;
        float* dp = pc.D + (t * nrows + k) * hc;
        const float z = (((dp[0] + dp[1]) + dp[2]) + dp[3]) / float(hc);
        g_lo3[t][k] = z / (1.0f + exp(-z));
    }
    GroupMemoryBarrierWithGroupSync();
    const int c = sg >> 1, dl = 8 * (sg & 1) + lane / 4, li = lane & 3;
    const int d = int(tg.x) * 16 + dl;
    uint8_t* w = pc.w_up + uint64_t(c * n + d) * a.rb_up;
    const int U = lr / UNIT;
    float acc[8];
    [ForceUnroll] for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int u = li; u < U; u += 4) {
        float q[MAXU];
        float4 sc;
        Dec_run<F>(w, u, q, sc);
        for (int t = 0; t < GR_NTOK; ++t) {
            float xs[MAXU], xsum[2] = {0.0f, 0.0f};
            [unroll] for (int e = 0; e < f_UNIT(F); ++e) xs[e] = g_lo3[t][UNIT * u + e];
            acc[t] += unit_combine(UNIT, q, sc, xs, xsum);
        }
    }
    for (int t = 0; t < GR_NTOK; ++t) {
        const float v = lane_sum<4>(acc[t]);
        if (li == 0) g_gate[t][c][dl] = 1.0f / (1.0f + exp(-v));
    }
    GroupMemoryBarrierWithGroupSync();
    const int t = tid / 16, j = tid & 15;
    if (t < GR_NTOK) {
        const int dd = int(tg.x) * 16 + j;
        float m = 0.0f;
        for (int cc = 0; cc < hc; ++cc) {
            const float xn = pc.Rn[(t * hc + cc) * n + dd] * pc.wn[cc * n + dd] * pc.rs[t * hc + cc];
            m = fma(xn, g_gate[t][cc][j], m);
        }
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
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += g_msum[t][k]; s1 += g_msum[t][k + 1]; }
        pc.xp[4 * a.n_tok * uint(n) + t * (n / 16) + tg.x] = s0 + s1;
    }
}

struct GrNormCombPush {
    float* R;
    float* ye;
    float* inj;
    float* w_norm;
    float* xn;
    GrArgs a;
    int* plan;
    float* ysh;
    float* sgv;
    uint topk;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gr_norm_comb(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrNormCombPush pc) {
    const GrArgs a = pc.a;
    const uint c = tg.x, t = tg.y, n = a.n_embd, topk = pc.topk;
    float* r = pc.R + (t * a.hc + c) * n;
    const float g = a.apply != 0 ? 2.0f / (1.0f + exp(-pc.inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    const float gs = 1.0f / (1.0f + exp(-pc.sgv[t]));
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        float v = r[d];
        if (a.apply != 0) {
            float s = 0.0f;
            for (uint j = 0; j < topk; ++j) s += pc.ye[pc.plan[P_TOKENT + t * topk + j] * int(n) + d];
            const float bo = fma(gs, pc.ysh[t * n + d], s);
            v = fma(bo, g, v);
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

#define GD3(F, T, R)                                                                                                  \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void gr_down3_##F##_##T##_##R(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrDown3Push pc) {         \
        gr_down3<F, T, R>(pc, tg, tid);                                                                              \
    }
GD3(30, 1, 1) GD3(30, 2, 1) GD3(30, 3, 1) GD3(30, 1, 2) GD3(30, 2, 2) GD3(30, 3, 2)
GD3(8, 1, 1) GD3(8, 2, 1) GD3(8, 3, 1) GD3(8, 1, 2) GD3(8, 2, 2) GD3(8, 3, 2)
#define GD3B(T, R)                                                                                                    \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void gr_down3b_##T##_##R(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrDown3Push pc) {              \
        gr_down3b<T, R>(pc, tg, tid);                                                                                \
    }
GD3B(1, 1) GD3B(2, 1) GD3B(3, 1) GD3B(1, 2) GD3B(2, 2) GD3B(3, 2)
[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void gr_up3_30(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrUp3Push pc) { gr_up3<30>(pc, tg, tid); }
[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void gr_up3_8(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrUp3Push pc) { gr_up3<8>(pc, tg, tid); }
