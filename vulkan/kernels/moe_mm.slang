// vulkan/kernels/moe_mm.slang - metal/kernels/moe_mm.metal in Slang (after quant/units/mma): the routed experts for a
// PROMPT chunk as grouped GEMMs, the v2 kernels (the M1-M4 path of the Mac engine; v3 is the M5's cooperative-tensor
// layout).  Plan (deterministic, three passes): the chunk's T x topk (token, expert) entries sorted by expert, then
// by entry index; each expert's run padded to a multiple of TM rows; tile i = (expert, first padded row).
//   mm_hist / mm_scan / mm_place / mm_clear   the plan
//   mm_gather   XS[row][k] = half(x[srt_tok[row]][k]) (0 on padding rows)
//   mm_gu2      HS[row][f] = half(silu(x . gate_f) * (x . up_f))
//   mm_down2    YE[row][d] = h . down_d (fp32)
//   mm_combine  y[t][d] = sum_k w(t, k) * YE[pos(t, k)][d], k in order
// The Mac's GEMMs are matmul2d (Metal 4); here each subgroup multiplies 16x16 fragments of the TM x TN tile on
// VK_KHR_cooperative_matrix (f16 operands, f32 accumulators): A = the gathered rows from device memory, B = the
// tile's TN x TK weights dequantized into threadgroup memory per K step.

struct MmPlanArgs {
    uint n_ent;
    uint topk;
    uint n_expert;
    uint tm;
    uint n_blocks;
};

groupshared uint mm_h[1024];
struct HistPush { int* ids; uint* bcnt; MmPlanArgs a; };
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mm_hist(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform HistPush pc) {
    const MmPlanArgs a = pc.a;
    for (uint e = tid; e < a.n_expert; e += 1024) mm_h[e] = 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint i = tg.x * 1024 + tid;
    if (i < a.n_ent) InterlockedAdd(mm_h[pc.ids[i]], 1u);
    GroupMemoryBarrierWithGroupSync();
    for (uint e = tid; e < a.n_expert; e += 1024) pc.bcnt[tg.x * a.n_expert + e] = mm_h[e];
}

groupshared uint mm_srows[32];
groupshared uint mm_stiles[32];
struct ScanPush { uint* bcnt; uint* boff; uint* eoff; int2* tiles; uint* meta; MmPlanArgs a; };
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mm_scan(uint tid : SV_GroupIndex, uniform ScanPush pc) {
    const MmPlanArgs a = pc.a;
    const uint sg = tid / 32, lane = WaveGetLaneIndex();
    if (tid < 32) { mm_srows[tid] = 0; mm_stiles[tid] = 0; }
    GroupMemoryBarrierWithGroupSync();
    const uint e = tid;
    uint tot = 0;
    if (e < a.n_expert)
        for (uint b = 0; b < a.n_blocks; ++b) {
            pc.boff[b * a.n_expert + e] = tot;
            tot += pc.bcnt[b * a.n_expert + e];
        }
    const uint nt = (tot + a.tm - 1) / a.tm, rows = nt * a.tm;
    const uint pr = WavePrefixSum(rows), pt = WavePrefixSum(nt);
    if (lane == 31) { mm_srows[sg] = pr + rows; mm_stiles[sg] = pt + nt; }
    GroupMemoryBarrierWithGroupSync();
    uint orow = 0, otile = 0, all_rows = 0, all_tiles = 0;
    for (uint i = 0; i < 32; ++i) {
        if (i < sg) { orow += mm_srows[i]; otile += mm_stiles[i]; }
        all_rows += mm_srows[i];
        all_tiles += mm_stiles[i];
    }
    if (e < a.n_expert) {
        const uint r0 = orow + pr, t0 = otile + pt;
        pc.eoff[e] = r0;
        for (uint k = 0; k < nt; ++k) pc.tiles[t0 + k] = int2(int(e), int(r0 + k * a.tm));
    }
    if (tid == 0) {
        pc.meta[0] = all_tiles;
        pc.meta[1] = all_rows;
        pc.meta[3] = all_tiles;
        pc.meta[4] = 1;
        pc.meta[6] = all_tiles;
        pc.meta[7] = 1;
        pc.meta[8] = all_rows;
        pc.meta[9] = 1;
        pc.meta[10] = 1;
    }
}

groupshared int mm_sid[1024];
struct PlacePush { int* ids; float* w; uint* boff; uint* eoff; int* srt_tok; float* srt_w; int* pos; MmPlanArgs a; };
// grid (n_blocks), 1024 threads.  srt_* cleared to the padding value beforehand by mm_clear.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mm_place(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform PlacePush pc) {
    const MmPlanArgs a = pc.a;
    const uint i = tg.x * 1024 + tid;
    const int e = i < a.n_ent ? pc.ids[i] : -1;
    mm_sid[tid] = e;
    GroupMemoryBarrierWithGroupSync();
    if (i >= a.n_ent) return;
    uint rank = 0;
    for (uint j = 0; j < tid; ++j) rank += mm_sid[j] == e ? 1u : 0u;
    const uint row = pc.eoff[e] + pc.boff[tg.x * a.n_expert + uint(e)] + rank;
    pc.srt_tok[row] = int(i / a.topk);
    pc.srt_w[row] = pc.w[i];
    pc.pos[i] = int(row);
}

struct ClearPush { int* srt_tok; uint n; };
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mm_clear(uint gid : SV_DispatchThreadID, uniform ClearPush pc) {
    if (gid < pc.n) pc.srt_tok[gid] = -1;
}

struct GatherPush { float* x; int* srt_tok; half* XS; uint n; };
// grid (rows), 256 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mm_gather(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GatherPush pc) {
    const int t = pc.srt_tok[tg.x];
    half* o = pc.XS + uint64_t(tg.x) * pc.n;
    for (uint k = tid; k < pc.n; k += 256) o[k] = t >= 0 ? half(pc.x[uint64_t(t) * pc.n + k]) : half(0.0f);
}

struct CombPush { float* YE; int* pos; float* srt_w; float* y; uint4 a; };   // a = (T, n_embd, topk, -)
// grid (T * n_embd / 256), 256 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mm_combine(uint gid : SV_DispatchThreadID, uniform CombPush pc) {
    const uint4 a = pc.a;
    const uint t = gid / a.y, d = gid % a.y;
    if (t >= a.x) return;
    float s = 0.0f;
    for (uint k = 0; k < a.z; ++k) {
        const int r = pc.pos[t * a.z + k];
        s = fma(pc.srt_w[r], pc.YE[uint64_t(r) * a.y + d], s);
    }
    pc.y[gid] = s;
}

// ---- the GEMMs -----------------------------------------------------------------------------------------------------
struct MmArgs {
    uint64_t gate_off;
    uint64_t up_off;
    uint64_t down_off;
    uint64_t gu_stride;
    uint64_t d_stride;
    uint gu_row;
    uint d_row;
    uint n_embd;
    uint n_ff;
};
struct MmPush {
    uint8_t* W;
    half* X;      // XS (mm_gu2) / HS (mm_down2)
    void* O;      // HS (half, mm_gu2) / YE (float, mm_down2)
    int2* tiles;
    MmArgs a;
    uint* smap;
    uint8_t* W1;
    uint8_t* W2;
    uint8_t* W3;
};

#ifndef STRATA_ARENA_SLANG
#define STRATA_ARENA_SLANG
[vk::constant_id(20)] const bool kArena = false;
uint8_t* arena_slot(uint v, uint8_t* W, uint8_t* W1, uint8_t* W2, uint8_t* W3) {
    const uint c = v >> 28;
    uint8_t* b = c == 0 ? W : (c == 1 ? W1 : (c == 2 ? W2 : W3));
    return b + uint64_t(v & 0x0FFFFFFFu) * 16384ull;
}
#endif
// a probe for the bench (function constant 5): 1 = the weights not decoded (a cheap fill), 2 = no matrix multiply
[vk::constant_id(5)] const int MM_PROBE = 0;

typealias MHA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MHB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MFC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
typealias MHC = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared half mm_wg[4096];
groupshared half mm_wu[4096];

// TN x TK weights of row_bytes-strided rows into threadgroup half [n][k]: thread-strided units of the format
void stage_w<let F : int, let TN : int, let TK : int>(uint8_t* w0, uint row_bytes, int k0, bool up, uint tid, uint nth) {
    const int UNIT = f_UNIT(F), UPC = TK / UNIT, H = UNIT / 2;
    for (int it = int(tid); it < TN * UPC; it += int(nth)) {
        const int r = it / UPC, uu = it - r * UPC;
        float wv[MAXU];
        int p0, p1;
        DW_run<F>(w0 + uint64_t(r) * row_bytes, k0 / UNIT + uu, k0, wv, p0, p1);
        [unroll]
        for (int j = 0; j < f_UNIT(F) / 2; ++j) {
            if (up) { mm_wu[r * TK + p0 + j] = half(wv[j]); mm_wu[r * TK + p1 + j] = half(wv[H + j]); }
            else { mm_wg[r * TK + p0 + j] = half(wv[j]); mm_wg[r * TK + p1 + j] = half(wv[H + j]); }
        }
    }
}

void mm_gu2<let F : int, let TM : int, let TN : int, let TK : int, let SG : int>(MmPush pc, uint3 tg, uint tid) {
    const int NTH = SG * 32, NFN = TN / 16, NF = (TM / 16) * NFN, FPS = (NF + SG - 1) / SG;
    load_tab<F>(tid, NTH);
    const MmArgs a = pc.a;
    const int sg = int(tid) / 32;
    const int2 tl = pc.tiles[tg.y];
    const int e = tl.x, row0 = tl.y, f0 = int(tg.x) * TN;
    uint8_t* E = kArena ? arena_slot(pc.smap[e], pc.W, pc.W1, pc.W2, pc.W3) : pc.W + uint64_t(e) * a.gu_stride;
    uint8_t* g0 = E + a.gate_off + uint64_t(f0) * a.gu_row;
    uint8_t* u0 = E + a.up_off + uint64_t(f0) * a.gu_row;
    MFC accg[FPS], accu[FPS];
    [ForceUnroll] for (int i = 0; i < FPS; ++i) { accg[i] = MFC(0.0f); accu[i] = MFC(0.0f); }
    GroupMemoryBarrierWithGroupSync();
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        if (MM_PROBE == 1) {
            for (int it = int(tid); it < TN * TK; it += NTH) { mm_wg[it] = half(k0 & 7); mm_wu[it] = half(it & 7); }
        } else {
            stage_w<F, TN, TK>(g0, a.gu_row, k0, false, tid, NTH);
            stage_w<F, TN, TK>(u0, a.gu_row, k0, true, tid, NTH);
        }
        GroupMemoryBarrierWithGroupSync();
        if (MM_PROBE != 2)
            [ForceUnroll]
            for (int kk = 0; kk < TK; kk += 16)
                [ForceUnroll]
                for (int i = 0; i < FPS; ++i) {
                    const int f = sg + SG * i;
                    if (f < NF) {
                        const int fm = f / NFN, fn = f % NFN;
                        const MHA A = MHA.Load<CoopMatMatrixLayout.RowMajor>(pc.X, uint((row0 + 16 * fm) * int(a.n_embd) + k0 + kk), a.n_embd);
                        const MHB Bg = MHB.Load<CoopMatMatrixLayout.ColumnMajor>(mm_wg, uint(16 * fn * TK + kk), TK);
                        const MHB Bu = MHB.Load<CoopMatMatrixLayout.ColumnMajor>(mm_wu, uint(16 * fn * TK + kk), TK);
                        accg[i] = coopMatMulAdd<float, false>(A, Bg, accg[i]);
                        accu[i] = coopMatMulAdd<float, false>(A, Bu, accu[i]);
                    }
                }
        GroupMemoryBarrierWithGroupSync();
    }
    half* HS = (half*)pc.O;
    [ForceUnroll]
    for (int i = 0; i < FPS; ++i) {
        const int f = sg + SG * i;
        if (f < NF) {
            const int fm = f / NFN, fn = f % NFN;
            for (uint k = 0; k < accg[i].GetLength(); ++k) {   // the same layout: element k of both is one (row, column)
                const float g = accg[i][k];
                accg[i][k] = g / (1.0f + exp(-g)) * accu[i][k];
            }
            const MHC h = MHC(accg[i]);
            h.Store<CoopMatMatrixLayout.RowMajor>(HS + uint64_t(row0 + 16 * fm) * a.n_ff + uint(f0 + 16 * fn), 0, a.n_ff);
        }
    }
}

void mm_down2<let F : int, let TM : int, let TN : int, let TK : int, let SG : int>(MmPush pc, uint3 tg, uint tid) {
    const int NTH = SG * 32, NFN = TN / 16, NF = (TM / 16) * NFN, FPS = (NF + SG - 1) / SG;
    load_tab<F>(tid, NTH);
    const MmArgs a = pc.a;
    const int sg = int(tid) / 32;
    const int2 tl = pc.tiles[tg.y];
    const int e = tl.x, row0 = tl.y, d0 = int(tg.x) * TN;
    uint8_t* w0 = (kArena ? arena_slot(pc.smap[e], pc.W, pc.W1, pc.W2, pc.W3) : pc.W + uint64_t(e) * a.d_stride) + a.down_off +
                  uint64_t(d0) * a.d_row;
    MFC acc[FPS];
    [ForceUnroll] for (int i = 0; i < FPS; ++i) acc[i] = MFC(0.0f);
    GroupMemoryBarrierWithGroupSync();
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        stage_w<F, TN, TK>(w0, a.d_row, k0, false, tid, NTH);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (int kk = 0; kk < TK; kk += 16)
            [ForceUnroll]
            for (int i = 0; i < FPS; ++i) {
                const int f = sg + SG * i;
                if (f < NF) {
                    const int fm = f / NFN, fn = f % NFN;
                    const MHA A = MHA.Load<CoopMatMatrixLayout.RowMajor>(pc.X, uint((row0 + 16 * fm) * int(a.n_ff) + k0 + kk), a.n_ff);
                    const MHB B = MHB.Load<CoopMatMatrixLayout.ColumnMajor>(mm_wg, uint(16 * fn * TK + kk), TK);
                    acc[i] = coopMatMulAdd<float, false>(A, B, acc[i]);
                }
            }
        GroupMemoryBarrierWithGroupSync();
    }
    float* YE = (float*)pc.O;
    [ForceUnroll]
    for (int i = 0; i < FPS; ++i) {
        const int f = sg + SG * i;
        if (f < NF) {
            const int fm = f / NFN, fn = f % NFN;
            acc[i].Store<CoopMatMatrixLayout.RowMajor>(YE + uint64_t(row0 + 16 * fm) * a.n_embd + uint(d0 + 16 * fn), 0, a.n_embd);
        }
    }
}

#define MMGU2(F, M, N, K, S) [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)] \
    void mm_gu2_##F##_##M##_##N##_##K##_##S(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform MmPush pc) { mm_gu2<F, M, N, K, S>(pc, tg, tid); }
#define MMDN2(F, M, N, K, S) [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)] \
    void mm_down2_##F##_##M##_##N##_##K##_##S(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform MmPush pc) { mm_down2<F, M, N, K, S>(pc, tg, tid); }
#define MM_GU2_SET(F) MMGU2(F, 32, 64, 32, 2) MMGU2(F, 32, 32, 32, 2) MMGU2(F, 32, 64, 16, 2) MMGU2(F, 64, 32, 16, 2) MMGU2(F, 16, 32, 32, 2) MMGU2(F, 16, 64, 16, 2) MMGU2(F, 16, 128, 32, 4) MMGU2(F, 32, 64, 16, 4) MMGU2(F, 16, 32, 32, 1) MMGU2(F, 16, 64, 32, 4) MMGU2(F, 16, 64, 64, 2) MMGU2(F, 32, 64, 64, 4) MMGU2(F, 64, 64, 64, 4) MMGU2(F, 16, 32, 64, 2) MMGU2(F, 32, 64, 32, 4) MMGU2(F, 64, 64, 32, 4) MMGU2(F, 16, 64, 32, 2)
#define MM_DN2_SET(F) MMDN2(F, 32, 64, 32, 2) MMDN2(F, 32, 32, 32, 2) MMDN2(F, 32, 64, 16, 2) MMDN2(F, 64, 32, 16, 2) MMDN2(F, 16, 32, 32, 2) MMDN2(F, 16, 64, 16, 2) MMDN2(F, 16, 128, 32, 4) MMDN2(F, 32, 64, 16, 4) MMDN2(F, 16, 32, 32, 1) MMDN2(F, 16, 64, 32, 4) MMDN2(F, 16, 64, 64, 2) MMDN2(F, 32, 64, 64, 4) MMDN2(F, 64, 64, 64, 4) MMDN2(F, 16, 32, 64, 2) MMDN2(F, 32, 64, 32, 4) MMDN2(F, 64, 64, 32, 4) MMDN2(F, 16, 64, 32, 2)
#define MM_GU2_U32(F) MMGU2(F, 16, 64, 32, 2) MMGU2(F, 16, 64, 64, 2) MMGU2(F, 32, 64, 32, 2) MMGU2(F, 16, 32, 32, 2) MMGU2(F, 64, 64, 32, 4)
#define MM_DN2_U32(F) MMDN2(F, 16, 64, 32, 2) MMDN2(F, 16, 64, 64, 2) MMDN2(F, 16, 32, 32, 2) MMDN2(F, 32, 32, 32, 2) MMDN2(F, 16, 128, 32, 4) MMDN2(F, 32, 64, 32, 2) MMDN2(F, 64, 64, 32, 4)
#define MM_GU2_KQ(F) MMGU2(F, 16, 32, 128, 2) MMGU2(F, 16, 16, 128, 2) MMGU2(F, 32, 32, 128, 2) MMGU2(F, 16, 32, 128, 4)
// T_Q4_K 12, T_Q5_K 13, T_Q6_K 14, T_Q8_0 8, T_Q5_1 7, T_IQ4_XS 23, T_Q2_0 42, T_IQ3_XXS 18, T_IQ2_XXS 16, T_IQ2_XS 17,
// T_IQ2_S 22, T_IQ3_S 21, T_IQ4_NL 20
MM_GU2_KQ(12) MM_GU2_KQ(13) MM_GU2_KQ(14) MM_GU2_U32(8) MM_DN2_U32(7) MM_DN2_U32(8)
MMGU2(23, 16, 64, 32, 2) MMGU2(23, 16, 64, 64, 2) MMGU2(23, 32, 64, 32, 2) MMGU2(23, 16, 32, 32, 2) MMGU2(23, 64, 64, 32, 4)
MM_GU2_SET(42) MM_GU2_SET(18) MM_GU2_SET(16) MM_GU2_SET(17) MM_GU2_SET(22) MM_GU2_SET(21)
MM_DN2_SET(42)
MMDN2(20, 16, 64, 32, 2) MMDN2(20, 16, 64, 64, 2) MMDN2(20, 16, 32, 32, 2) MMDN2(20, 32, 32, 32, 2) MMDN2(20, 16, 128, 32, 4)
