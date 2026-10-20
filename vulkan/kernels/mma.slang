// vulkan/kernels/mma.slang - metal/kernels/mma.metal in Slang (after quant/units): a GEMV for windows of tokens on the
// matrix units.  y[t][r] = W[r] . x[t] for NTOK (1..8) tokens: each simdgroup owns 8 rows; the K axis runs in chunks of
// 128.  Per chunk the simdgroup decodes its 8 x 128 weights - exactly the dequantized weights (ref_quant's arithmetic)
// - into threadgroup memory, then 16 multiply-accumulates of 8 x 8 tiles (VK_KHR_cooperative_matrix, 8x8x8 f32 at
// subgroup scope on Honeykrisp: Metal's simdgroup_float8x8): A = the weight tile, B = the 8 (k) x 8 (token) slice of x
// loaded column-major from device memory (x has 8 rows; the tokens past the window are padding).
// Activations are the raw plane; n_in must be a multiple of 128.

using namespace linalg;
typealias Mat8A = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixA>;
typealias Mat8B = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixB>;
typealias Mat8C = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixAccumulator>;

// ---- decoders to actual weights: w[UNIT] and the chunk offsets of its two halves (p0: w[0..H), p1: w[H..UNIT)) ----
void DW_run<let F : int>(uint8_t* row, int u, int k0, inout float w[MAXU], out int p0, out int p1) {
    const int UNIT = f_UNIT(F);
    if (F == T_Q5_1) {
        uint8_t* b = row + u * 24;
        float q[MAXU];
        const float d = deq<T_Q5_1>(b, 0, q), m = rdh(b + 2);
        [ForceUnroll] for (int j = 0; j < 32; ++j) w[j] = q[j] * d + m;
        p0 = 32 * u - k0;
        p1 = p0 + 16;
    } else if (F == T_F32) {
        float* p = (float*)(row + 64 * u);
        [ForceUnroll] for (int j = 0; j < 16; ++j) w[j] = p[j];
        p0 = 16 * u - k0;
        p1 = p0 + 8;
    } else if (F == T_BF16) {
        uint4* p = (uint4*)(row + 32 * u);
        const uint4 a = loadAligned<16>(p), c = loadAligned<16>(p + 1);
        const uint v[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
        [ForceUnroll]
        for (int k = 0; k < 8; ++k) {
            w[2 * k] = asfloat(v[k] << 16);
            w[2 * k + 1] = asfloat(v[k] & 0xFFFF0000u);
        }
        p0 = 16 * u - k0;
        p1 = p0 + 8;
    } else if (F == T_Q4_K || F == T_Q5_K) {
        // d1 * code - m1 (ref: d * sc, dmin * m), runs at 256b + 64c + 16m and + 32
        uint8_t* b = row + (u >> 3) * (F == T_Q4_K ? 144 : 176);
        const int c = (u >> 1) & 3, m = u & 1;
        const float d = rdh(b), dm = rdh(b + 2);
        float s1, m1, s2, m2;
        scale_min_k4(2 * c, b + 4, s1, m1);
        scale_min_k4(2 * c + 1, b + 4, s2, m2);
        const float d1 = d * s1, mm1 = dm * m1, d2 = d * s2, mm2 = dm * m2;
        if (F == T_Q4_K) {
            uint* q = (uint*)(b + 16 + 32 * c + 16 * m);   // 16-byte aligned (144-byte blocks)
            [ForceUnroll]
            for (int i = 0; i < 4; ++i) {
                const uint v = q[i];
                [ForceUnroll]
                for (int e = 0; e < 4; ++e) {
                    const uint by = (v >> (8 * e)) & 255u;
                    w[4 * i + e] = d1 * float(by & 15u) - mm1;
                    w[16 + 4 * i + e] = d2 * float(by >> 4) - mm2;
                }
            }
        } else {
            uint* ql = (uint*)(b + 48 + 32 * c + 16 * m);   // 16-byte aligned (176-byte blocks)
            uint* qh = (uint*)(b + 16 + 16 * m);
            const uint u1 = 1u << (2 * c), u2 = 2u << (2 * c);
            [ForceUnroll]
            for (int i = 0; i < 4; ++i) {
                const uint v = ql[i], h = qh[i];
                [ForceUnroll]
                for (int e = 0; e < 4; ++e) {
                    const uint by = (v >> (8 * e)) & 255u, hb = (h >> (8 * e)) & 255u;
                    w[4 * i + e] = d1 * float((by & 15u) + ((hb & u1) != 0 ? 16u : 0u)) - mm1;
                    w[16 + 4 * i + e] = d2 * float((by >> 4) + ((hb & u2) != 0 ? 16u : 0u)) - mm2;
                }
            }
        }
        p0 = 256 * (u >> 3) + 64 * c + 16 * m - k0;
        p1 = p0 + 32;
    } else if (F == T_Q6_K) {
        // d * sc * (code - 32), runs at 256b + 128n + 32p + 16m and + 64
        uint8_t* b = row + (u >> 3) * 210;
        const int n = (u >> 2) & 1, m = (u >> 1) & 1, p = u & 1;
        uint16_t* ql = (uint16_t*)(b + 64 * n + 32 * p + 16 * m);   // 2-byte aligned (210-byte blocks): pairs
        uint16_t* qh = (uint16_t*)(b + 128 + 32 * n + 16 * m);
        int8_t* sc = (int8_t*)(b + 192);
        const float d = rdh(b + 208);
        const float ds1 = d * float(sc[8 * n + m + 2 * p]), ds2 = d * float(sc[8 * n + m + 2 * p + 4]);
        [ForceUnroll]
        for (int i = 0; i < 4; ++i) {
            const uint v = rd2x2(ql, 2 * i), h = rd2x2(qh, 2 * i);
            [ForceUnroll]
            for (int e = 0; e < 4; ++e) {
                const uint by = (v >> (8 * e)) & 255u, hb = (h >> (8 * e)) & 255u;
                const int c1 = int((by & 15u) | (((hb >> (2 * p)) & 3u) << 4)) - 32;
                const int c2 = int((by >> 4) | (((hb >> (2 * p + 4)) & 3u) << 4)) - 32;
                w[4 * i + e] = ds1 * float(c1);
                w[16 + 4 * i + e] = ds2 * float(c2);
            }
        }
        p0 = 256 * (u >> 3) + 128 * n + 32 * p + 16 * m - k0;
        p1 = p0 + 64;
    } else if (F == T_Q3_K) {
        // d_all * (sc - 32) * (code - (hbit ? 0 : 4)), contiguous 32
        uint8_t* b = row + (u >> 3) * 110;
        const int n = (u >> 2) & 1, j = u & 3;
        uint8_t* q = b + 32 + 32 * n;
        uint8_t* hm = b;
        uint8_t* s = b + 96;
        const int i1 = 8 * n + 2 * j, i2 = i1 + 1;
        const int l1 = int((i1 < 8 ? uint(s[i1]) & 0xFu : uint(s[i1 - 8]) >> 4) | (((uint(s[8 + (i1 & 3)]) >> (2 * (i1 >> 2))) & 3u) << 4));
        const int l2 = int((i2 < 8 ? uint(s[i2]) & 0xFu : uint(s[i2 - 8]) >> 4) | (((uint(s[8 + (i2 & 3)]) >> (2 * (i2 >> 2))) & 3u) << 4));
        const float d = rdh(b + 108);
        const float dl1 = d * float(l1 - 32), dl2 = d * float(l2 - 32);
        const uint hb = 1u << (4 * n + j);
        [ForceUnroll]
        for (int l = 0; l < 16; ++l) {
            w[l] = dl1 * float(int((uint(q[l]) >> (2 * j)) & 3u) - ((uint(hm[l]) & hb) != 0 ? 0 : 4));
            w[16 + l] = dl2 * float(int((uint(q[l + 16]) >> (2 * j)) & 3u) - ((uint(hm[l + 16]) & hb) != 0 ? 0 : 4));
        }
        p0 = 32 * u - k0;
        p1 = p0 + 16;
    } else {
        // generic: w = s * q (deq is ref_quant's order: the scale times the code), contiguous
        const int UPB = f_QK(F) / UNIT;
        float q[MAXU];
        const float s = deq<F>(row + (u / UPB) * f_BB(F), u % UPB, q);
        [unroll] for (int j = 0; j < f_UNIT(F); ++j) w[j] = q[j] * s;
        p0 = u * UNIT - k0;
        p1 = p0 + UNIT / 2;
    }
}

struct MmaArgs {
    uint n_in;
    uint n_out;
    uint row_bytes;
    uint x_stride;
    uint y_stride;
};
struct MmaPush {
    uint8_t* W;
    float* x;
    float* y;
    MmaArgs a;
};
[vk::constant_id(4)] const int MMA_NTOK = 1;

groupshared float g_mwt[4 * 8 * 128];
groupshared float g_mct[4 * 64];

void gemv_mma<let F : int, let SG : int>(MmaPush pc, uint3 tg, uint tid) {
    const int UNIT = f_UNIT(F), UPC = 128 / UNIT, H = UNIT / 2;
    const MmaArgs a = pc.a;
    load_tab<F>(tid, SG * 32);
    GroupMemoryBarrierWithGroupSync();
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int row0 = (int(tg.x) * SG + sg) * 8;
    Mat8C C = Mat8C(0.0f);
    const int wb = sg * 8 * 128;
    for (int k0 = 0; k0 < int(a.n_in); k0 += 128) {
        for (int it = lane; it < 8 * UPC; it += 32) {
            const int r = it / UPC, uu = it - r * UPC;
            const int rr = min(row0 + r, int(a.n_out) - 1);
            float w[MAXU];
            int p0, p1;
            DW_run<F>(pc.W + uint64_t(rr) * a.row_bytes, k0 / UNIT + uu, k0, w, p0, p1);
            [unroll]
            for (int j = 0; j < f_UNIT(F) / 2; ++j) {
                g_mwt[wb + r * 128 + p0 + j] = w[j];
                g_mwt[wb + r * 128 + p1 + j] = w[H + j];
            }
        }
        GroupMemoryBarrierWithWaveSync();
        [ForceUnroll]
        for (int k8 = 0; k8 < 128; k8 += 8) {
            const Mat8A A = Mat8A.Load<CoopMatMatrixLayout.RowMajor>(g_mwt, uint(wb + k8), 128);
            const Mat8B B = Mat8B.Load<CoopMatMatrixLayout.ColumnMajor>(pc.x, uint(k0 + k8), a.x_stride);
            C = coopMatMulAdd<float, false>(A, B, C);
        }
        GroupMemoryBarrierWithWaveSync();
    }
    C.Store<CoopMatMatrixLayout.RowMajor>(g_mct, uint(sg * 64), 8);
    GroupMemoryBarrierWithWaveSync();
    for (int e = lane; e < 64; e += 32) {
        const int r = e >> 3, t = e & 7;
        if (t < MMA_NTOK && row0 + r < int(a.n_out)) pc.y[t * int(a.y_stride) + row0 + r] = g_mct[sg * 64 + e];
    }
}

#define MM(F, S)                                                                                                      \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void gemv_mma_##F##_##S(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform MmaPush pc) {                   \
        gemv_mma<F, S>(pc, tg, tid);                                                                                 \
    }
#define MM_SET(F) MM(F, 1) MM(F, 2) MM(F, 4)
// T_BF16 30, T_Q8_0 8, T_Q4_0 2, T_Q5_0 6, T_IQ4_NL 20, T_IQ4_XS 23, T_Q2_0 42, T_IQ3_S 21, T_Q3_K 11, T_Q4_K 12,
// T_Q5_K 13, T_Q6_K 14
MM_SET(30) MM_SET(8) MM_SET(2) MM_SET(6) MM_SET(20) MM_SET(23) MM_SET(42) MM_SET(21) MM_SET(11) MM_SET(12) MM_SET(13) MM_SET(14)
