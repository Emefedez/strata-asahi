// vulkan/kernels/prompt.slang - metal/kernels/prompt.metal in Slang (after quant/units/mma/dense_mm/gr): the
// elementwise steps of a prompt chunk around the GEMMs:
//   route_topk     per token: softmax over the 512 logits, top 10 (ties: the lower expert), weights renormalised
//                  over the selected with a 2^-14 floor - moe_route's rule; one simdgroup a token
//   gr_epi_down    after the HC down GEMM: lo_h = half(silu(v / hc)), inject = v
//   gr_epi_up      after the HC up GEMM: mixed = mean_c xn[c] * sigmoid(G[c]) -> float and half
//   swiglu_h       h = half(silu(g) * u)
//   moe_out        bo = y_routed + sigmoid(sg) * y_shared
//   gr_norm_h      gr_norm with xn written as fp16 only
//   gemm_hc_up     G = lo . w_up^T (w_up's rows interleaved by dq_f16 perm = hc) with the gr_epi_up epilogue fused:
//                  G is never written.  The Mac's is matmul2d; here the cooperative-matrix GEMM of dense_mm.slang.

struct RoutePush {
    float* logits;
    int* ids;
    float* w;
    uint n_tok;
};

// 8 simdgroups (256 threads) a threadgroup, one token each
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void route_topk(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform RoutePush pc) {
    const uint sg = tid / 32, lane = tid % 32;
    const uint t = tg.x * 8 + sg;
    if (t >= pc.n_tok) return;
    float* l = pc.logits + uint64_t(t) * 512;
    float v[16];
    [ForceUnroll] for (int i = 0; i < 16; ++i) v[i] = l[lane + 32 * i];
    float mx = -INFINITY;
    [ForceUnroll] for (int i = 0; i < 16; ++i) mx = max(mx, v[i]);
    for (uint m = 16; m > 0; m >>= 1) mx = max(mx, shfl_xor(mx, m));
    float sum = 0.0f;
    [ForceUnroll] for (int i = 0; i < 16; ++i) {
        v[i] = exp(v[i] - mx);
        sum += v[i];
    }
    for (uint m = 16; m > 0; m >>= 1) sum += shfl_xor(sum, m);
    const float rcp = 1.0f / sum;
    [ForceUnroll] for (int i = 0; i < 16; ++i) {
        v[i] *= rcp;
        if (isnan(v[i])) v[i] = -FLT_MAX;
    }
    float selected = 0.0f, ssum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = int(lane);
        [ForceUnroll] for (int i = 1; i < 16; ++i)
            if (v[i] > best) { best = v[i]; ex = int(lane) + 32 * i; }
        for (uint m = 16; m > 0; m >>= 1) {
            const float ob = shfl_xor(best, m);
            const int oe = shfl_xor(ex, m);
            if (ob > best || (ob == best && oe < ex)) { best = ob; ex = oe; }
        }
        if ((ex & 31) == int(lane)) {
            // v[ex / 32] = -INF with a register index known only at run time
            [ForceUnroll] for (int i = 0; i < 16; ++i)
                if (i == ex / 32) v[i] = -INFINITY;
            pc.ids[t * 10 + rank] = ex;
            ssum += best;
        }
        if (rank == int(lane)) selected = best;
    }
    for (uint m = 16; m > 0; m >>= 1) ssum += shfl_xor(ssum, m);
    ssum = max(ssum, 6.103515625e-5f);
    if (lane < 10) pc.w[t * 10 + lane] = selected * (1.0f / ssum);
}

struct EpiArgs {
    uint n_tok;
    uint n_embd;
    uint hc;
    uint hc_lr;
};

struct EpiDownPush {
    float* V;
    half* lo_h;
    float* inject;
    EpiArgs a;
};

// grid (T * (hc_lr + hc) / 256)
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gr_epi_down(uint gid : SV_DispatchThreadID, uniform EpiDownPush pc) {
    const EpiArgs a = pc.a;
    const uint nr = a.hc_lr + a.hc, t = gid / nr, r = gid % nr;
    if (t >= a.n_tok) return;
    const float v = pc.V[gid];
    if (r < a.hc_lr) {
        const float z = v / float(a.hc);
        pc.lo_h[t * a.hc_lr + r] = half(z / (1.0f + exp(-z)));
    } else {
        pc.inject[t * a.hc + r - a.hc_lr] = v;
    }
}

struct EpiUpPush {
    float* G;
    float* xn;
    float* mixed;
    half* mixed_h;
    EpiArgs a;
};

// grid (T * n_embd / 256)
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gr_epi_up(uint gid : SV_DispatchThreadID, uniform EpiUpPush pc) {
    const EpiArgs a = pc.a;
    const uint n = a.n_embd, t = gid / n, d = gid % n;
    if (t >= a.n_tok) return;
    float m = 0.0f;
    for (uint c = 0; c < a.hc; ++c) {
        const float g = pc.G[uint64_t(t) * a.hc * n + c * n + d];
        m = fma(pc.xn[uint64_t(t) * a.hc * n + c * n + d], 1.0f / (1.0f + exp(-g)), m);
    }
    m = m / float(a.hc);
    pc.mixed[gid] = m;
    pc.mixed_h[gid] = half(m);
}

struct SwigluHPush {
    float* g;
    float* u;
    half* h;
    uint n;
};

[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void swiglu_h(uint gid : SV_DispatchThreadID, uniform SwigluHPush pc) {
    if (gid >= pc.n) return;
    const float gv = pc.g[gid];
    pc.h[gid] = half(gv / (1.0f + exp(-gv)) * pc.u[gid]);
}

struct MoeOutPush {
    float* yr;
    float* ys;
    float* sg;
    float* bo;
    uint2 a;   // (n_tok, n_embd)
};

// grid (T * n_embd / 256); sg: T values
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void moe_out(uint gid : SV_DispatchThreadID, uniform MoeOutPush pc) {
    const uint t = gid / pc.a.y;
    if (t >= pc.a.x) return;
    pc.bo[gid] = fma(1.0f / (1.0f + exp(-pc.sg[t])), pc.ys[gid], pc.yr[gid]);
}

struct GrNormHPush {
    float* R;
    float* bo;
    float* inj;
    float* w_norm;
    half* xnh;
    GrArgs a;
};

groupshared float g_nh_part[8];

// one threadgroup per (stream c, token t), 256 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gr_norm_h(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GrNormHPush pc) {
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
    if (WaveGetLaneIndex() == 0) g_nh_part[tid / 32] = ss;
    GroupMemoryBarrierWithGroupSync();
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += g_nh_part[i];
    const float rs = rsqrt(tot / float(n) + a.eps);
    half* o = pc.xnh + (t * a.hc + c) * n;
    float* w = pc.w_norm + c * n;
    for (uint d = tid; d < n; d += 256) o[d] = half(r[d] * w[d] * rs);
}

// ---- gemm_hc_up --------------------------------------------------------------------------------------------------
struct HcUpPush {
    half* A;        // lo_h: M x K
    half* B;        // w_up (interleaved): N x K
    half* xnh;
    float* mixed;
    half* mixh;
    GemmArgs g;
    EpiArgs e;
};

// its own staging (64-row tiles at most) and the output tile: 4 + 4 + 16 KB
groupshared half g_ua[64 * GK];
groupshared half g_ub[64 * GK];
groupshared float g_uo[64 * 64];

void stage_up<let R : int>(half* src, int rows, int K, int r0, int k0, bool isA, uint tid) {
    for (int e = int(tid) * 8; e < R * GK; e += 128 * 8) {
        const int r = e / GK, c = e % GK;
        uint4 v = uint4(0, 0, 0, 0);
        if (r0 + r < rows && k0 + c < K) v = loadAligned<16>((uint4*)(src + uint64_t(r0 + r) * uint(K) + uint(k0 + c)));
        [ForceUnroll]
        for (int i = 0; i < 4; ++i) {
            const uint w = i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
            const half lo = asfloat16(uint16_t(w & 0xFFFFu)), hi = asfloat16(uint16_t(w >> 16));
            if (isA) { g_ua[e + 2 * i] = lo; g_ua[e + 2 * i + 1] = hi; }
            else { g_ub[e + 2 * i] = lo; g_ub[e + 2 * i + 1] = hi; }
        }
    }
}

// grid (hc * n / TN, ceil(T / TM)), 128 threads (4 subgroups as 2 x 2)
void gemm_hc_up<let TM : int, let TN : int>(HcUpPush pc, uint3 tg, uint tid) {
    const int WM = TM / 32, WN = TN / 32;
    const GemmArgs g = pc.g;
    const EpiArgs e = pc.e;
    const int sg = int(tid) / 32;
    const int sm = sg >> 1, sn = sg & 1;
    const int m0 = int(tg.y) * TM, n0 = int(tg.x) * TN;
    FMatC acc[WM][WN];
    [ForceUnroll] for (int i = 0; i < WM; ++i)
        [ForceUnroll] for (int j = 0; j < WN; ++j) acc[i][j] = FMatC(0.0f);
    for (int k0 = 0; k0 < g.K; k0 += GK) {
        stage_up<TM>(pc.A, g.M, g.K, m0, k0, true, tid);
        stage_up<TN>(pc.B, g.N, g.K, n0, k0, false, tid);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (int kk = 0; kk < GK; kk += 16) {
            HMatA fa[WM];
            HMatB fb[WN];
            [ForceUnroll] for (int i = 0; i < WM; ++i)
                fa[i] = HMatA.Load<CoopMatMatrixLayout.RowMajor>(g_ua, uint((sm * (TM / 2) + 16 * i) * GK + kk), GK);
            [ForceUnroll] for (int j = 0; j < WN; ++j)
                fb[j] = HMatB.Load<CoopMatMatrixLayout.ColumnMajor>(g_ub, uint((sn * (TN / 2) + 16 * j) * GK + kk), GK);
            [ForceUnroll] for (int i = 0; i < WM; ++i)
                [ForceUnroll] for (int j = 0; j < WN; ++j) acc[i][j] = coopMatMulAdd<float, false>(fa[i], fb[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // the tile into threadgroup memory (row m, column n: g_uo[m * TN + n]), then the epilogue
    [ForceUnroll] for (int i = 0; i < WM; ++i)
        [ForceUnroll] for (int j = 0; j < WN; ++j)
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(g_uo, uint((sm * (TM / 2) + 16 * i) * TN + sn * (TN / 2) + 16 * j), uint(TN));
    GroupMemoryBarrierWithGroupSync();
    const int hc = int(e.hc), n = int(e.n_embd), DPT = TN / hc;
    const int t0 = int(tg.y) * TM, d0 = int(tg.x) * DPT;
    for (int idx = int(tid); idx < TM * DPT; idx += 128) {
        const int m = idx / DPT, dd = idx - m * DPT, t = t0 + m;
        if (t >= int(e.n_tok)) continue;
        float s = 0.0f;
        for (int c = 0; c < hc; ++c)
            s = fma(float(pc.xnh[uint64_t(t) * uint(hc * n) + uint(c * n + d0 + dd)]), 1.0f / (1.0f + exp(-g_uo[m * TN + dd * hc + c])), s);
        s = s / float(hc);
        pc.mixed[uint64_t(t) * uint(n) + uint(d0 + dd)] = s;
        pc.mixh[uint64_t(t) * uint(n) + uint(d0 + dd)] = half(s);
    }
}

#define HCUP(M, N) [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)] \
    void gemm_hc_up_##M##_##N##_4(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform HcUpPush pc) { gemm_hc_up<M, N>(pc, tg, tid); }
HCUP(64, 64) HCUP(64, 32) HCUP(32, 64)
