// vulkan/kernels/dense_mm.slang - metal/kernels/dense_mm.metal in Slang (after quant/units/mma): dense projections
// for a PROMPT chunk:
//   act_f16    X[t][k] -> half (the matrix units' operand)
//   dq_f16<F>  a quantized matrix -> half rows (ref_quant's values rounded to fp16), once per chunk
//   gemm_f16   Y[t][n] = Xh[t] . Wh[n] (fp32), TM x TN tiles, K dynamic; rows past n_out / t past n_tok not written.
// The Mac's gemm_f16 is MetalPerformancePrimitives' matmul2d (Metal 4); here it is a tiled GEMM on
// VK_KHR_cooperative_matrix (16x16x16, f16 operands, f32 accumulators): 4 subgroups as 2 x 2, each (TM/2) x (TN/2)
// of the tile; K in steps of 32 staged through threadgroup memory (zeros past the extents).

struct ActPush {
    float* x;
    half* xh;
    uint n;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void act_f16(uint gid : SV_DispatchThreadID, uniform ActPush pc) {
    if (gid < pc.n) pc.xh[gid] = half(pc.x[gid]);
}

struct DqArgs {
    uint n_in;
    uint n_out;
    uint row_bytes;
    uint perm;   // 0: row r -> r; p > 0: row r = c * (n_out / p) + d -> d * p + c (gemm_hc_up's interleave)
};
struct DqPush {
    uint8_t* W;
    half* Wh;
    DqArgs a;
};

// one thread per unit: grid (units per row / 64, n_out), 64 threads
void dq_f16<let F : int>(DqPush pc, uint3 tg, uint tid) {
    const int UNIT = f_UNIT(F), H = UNIT / 2;
    load_tab<F>(tid, 64);
    GroupMemoryBarrierWithGroupSync();
    const DqArgs a = pc.a;
    const int u = int(tg.x) * 64 + int(tid), r = int(tg.y);
    if (u * UNIT >= int(a.n_in)) return;
    const int k0 = (u * UNIT) / 128 * 128;   // a 128-wide chunk holds whole units of every format
    float w[MAXU];
    int p0, p1;
    DW_run<F>(pc.W + uint64_t(r) * a.row_bytes, u, k0, w, p0, p1);
    const int q = a.perm != 0 ? int(a.n_out / a.perm) : 1;
    const int ro = a.perm != 0 ? (r % q) * int(a.perm) + r / q : r;
    half* o = pc.Wh + uint64_t(ro) * a.n_in + k0;
    [unroll]
    for (int j = 0; j < f_UNIT(F) / 2; ++j) {
        o[p0 + j] = half(w[j]);
        o[p1 + j] = half(w[H + j]);
    }
}

#define DQ(F) [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)] \
    void dq_f16_##F(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform DqPush pc) { dq_f16<F>(pc, tg, tid); }
// T_F32 0, T_BF16 30, T_Q8_0 8, T_Q4_0 2, T_Q5_0 6, T_IQ4_NL 20, T_IQ4_XS 23, T_Q2_0 42, T_IQ3_S 21, T_Q3_K 11,
// T_Q4_K 12, T_Q5_K 13, T_Q6_K 14
DQ(0) DQ(30) DQ(8) DQ(2) DQ(6) DQ(20) DQ(23) DQ(42) DQ(21) DQ(11) DQ(12) DQ(13) DQ(14)

// ---- the GEMM ------------------------------------------------------------------------------------------------------
typealias HMatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias HMatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FMatC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

struct GemmArgs { int M, N, K; int pad; };   // Y is M rows of N floats
struct GemmPush {
    half* A;     // M x K (row-major)
    half* B;     // N x K (row-major: C = A . B^T)
    float* Y;
    GemmArgs a;
};

#define GK 32
groupshared half g_ga[128 * GK];
groupshared half g_gb[128 * GK];
groupshared float g_gc[4 * 256];

// rows [r0, r0 + R) of a row-major matrix of `rows` rows and K columns, columns [k0, k0 + GK) -> tile[R][GK] (zeros
// past the extents); 128 threads, 8 halves (one 16-byte load) at a time
void stage<let R : int>(half* src, int rows, int K, int r0, int k0, bool isA, uint tid) {
    for (int e = int(tid) * 8; e < R * GK; e += 128 * 8) {
        const int r = e / GK, c = e % GK;
        uint4 v = uint4(0, 0, 0, 0);
        if (r0 + r < rows && k0 + c < K) v = loadAligned<16>((uint4*)(src + uint64_t(r0 + r) * uint(K) + uint(k0 + c)));
        // 8 halves as four 32-bit words
        [ForceUnroll]
        for (int i = 0; i < 4; ++i) {
            const uint w = i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
            const half lo = asfloat16(uint16_t(w & 0xFFFFu)), hi = asfloat16(uint16_t(w >> 16));
            if (isA) { g_ga[e + 2 * i] = lo; g_ga[e + 2 * i + 1] = hi; }
            else { g_gb[e + 2 * i] = lo; g_gb[e + 2 * i + 1] = hi; }
        }
    }
}

void gemm_f16<let TM : int, let TN : int>(GemmPush pc, uint3 tg, uint tid) {
    const int WM = TM / 32, WN = TN / 32;   // 16x16 fragments a subgroup holds along M and N
    const GemmArgs a = pc.a;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int sm = sg >> 1, sn = sg & 1;     // the subgroup's quarter of the tile
    const int m0 = int(tg.y) * TM, n0 = int(tg.x) * TN;
    FMatC acc[WM][WN];
    [ForceUnroll] for (int i = 0; i < WM; ++i)
        [ForceUnroll] for (int j = 0; j < WN; ++j) acc[i][j] = FMatC(0.0f);
    for (int k0 = 0; k0 < a.K; k0 += GK) {
        stage<TM>(pc.A, a.M, a.K, m0, k0, true, tid);
        stage<TN>(pc.B, a.N, a.K, n0, k0, false, tid);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (int kk = 0; kk < GK; kk += 16) {
            HMatA fa[WM];
            HMatB fb[WN];
            [ForceUnroll] for (int i = 0; i < WM; ++i)
                fa[i] = HMatA.Load<CoopMatMatrixLayout.RowMajor>(g_ga, uint((sm * (TM / 2) + 16 * i) * GK + kk), GK);
            [ForceUnroll] for (int j = 0; j < WN; ++j)   // B's 16 (k) x 16 (n) fragment: rows n of the [n][k] tile
                fb[j] = HMatB.Load<CoopMatMatrixLayout.ColumnMajor>(g_gb, uint((sn * (TN / 2) + 16 * j) * GK + kk), GK);
            [ForceUnroll] for (int i = 0; i < WM; ++i)
                [ForceUnroll] for (int j = 0; j < WN; ++j) acc[i][j] = coopMatMulAdd<float, false>(fa[i], fb[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // the epilogue: whole fragments straight to Y, edge fragments through threadgroup memory with the bounds
    [ForceUnroll]
    for (int i = 0; i < WM; ++i)
        [ForceUnroll]
        for (int j = 0; j < WN; ++j) {
            const int r0 = m0 + sm * (TM / 2) + 16 * i, c0 = n0 + sn * (TN / 2) + 16 * j;
            if (r0 + 16 <= a.M && c0 + 16 <= a.N) {
                acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(pc.Y + uint64_t(r0) * uint(a.N) + uint(c0), 0, uint(a.N));
            } else {
                acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(g_gc, uint(sg * 256), 16);
                GroupMemoryBarrierWithWaveSync();
                for (int e = lane; e < 256; e += 32) {
                    const int r = r0 + e / 16, c = c0 + e % 16;
                    if (r < a.M && c < a.N) pc.Y[uint64_t(r) * uint(a.N) + uint(c)] = g_gc[sg * 256 + e];
                }
                GroupMemoryBarrierWithWaveSync();
            }
        }
}

#define GEMM(M, N) [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)] \
    void gemm_f16_##M##_##N##_4(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GemmPush pc) { gemm_f16<M, N>(pc, tg, tid); }
GEMM(64, 64) GEMM(128, 64) GEMM(64, 128) GEMM(64, 32)
