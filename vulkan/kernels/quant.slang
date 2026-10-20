// vulkan/kernels/quant.slang - metal/kernels/quant.metal in Slang: the weight formats as "units" (UNIT consecutive
// weights of one row that one lane decodes at a time into raw values q[0..UNIT) and a scale s, so that the unit's dot
// with x is s * sum(q[j] * x[j])). Same decode rules, same tables (ggml_common.slang), same arithmetic.
//
// Differences from the MSL, none of which changes a value:
//   * the format is a value generic <let F : int> with compile-time branches (MSL: template specialisations);
//   * a unit is held in a 32-element array whatever the format (Slang cannot size an array from a generic); every
//     loop has a compile-time bound and unrolls, so only the elements a format uses are kept;
//   * the threadgroup tables are one global groupshared array per table (Slang cannot pass groupshared pointers);
//     a kernel allocates only the tables its format reads.

#define T_Q4_0 2
#define T_Q3_K 11
#define T_BF16 30
#define T_F32 0
#define T_Q5_0 6
#define T_Q5_1 7
#define T_Q8_0 8
#define T_Q4_K 12
#define T_Q5_K 13
#define T_Q6_K 14
#define T_IQ2_XXS 16
#define T_IQ2_XS 17
#define T_IQ3_XXS 18
#define T_IQ4_NL 20
#define T_IQ3_S 21
#define T_IQ2_S 22
#define T_IQ4_XS 23
#define T_Q2_0 42
#define T_Q2_0R 142   // Q2_0 read with raw activations (codes - 1 in the decoder; no prepared x needed)

#define MAXU 32       // the largest UNIT

// QK block elements, BB block bytes, UNIT weights per unit (MSL fmt<F>)
int f_QK(int F) {
    switch (F) {
        case T_Q2_0: case T_Q2_0R: return 64;
        case T_IQ4_NL: case T_Q8_0: case T_Q4_0: case T_Q5_0: case T_Q5_1: return 32;
        case T_BF16: case T_F32: return 16;
        default: return 256;
    }
}
int f_BB(int F) {
    switch (F) {
        case T_Q2_0: case T_Q2_0R: return 18;
        case T_IQ3_XXS: return 98;
        case T_IQ2_XXS: return 66;
        case T_IQ2_XS: return 74;
        case T_IQ2_S: return 82;
        case T_IQ3_S: return 110;
        case T_IQ4_NL: return 18;
        case T_IQ4_XS: return 136;
        case T_Q3_K: return 110;
        case T_Q4_K: return 144;
        case T_Q5_K: return 176;
        case T_Q6_K: return 210;
        case T_BF16: return 32;
        case T_F32: return 64;
        case T_Q8_0: return 34;
        case T_Q4_0: return 18;
        case T_Q5_0: return 22;
        case T_Q5_1: return 24;
        default: return 0;
    }
}
int f_UNIT(int F) {
    switch (F) {
        case T_Q2_0: case T_Q2_0R: case T_IQ3_XXS: case T_IQ2_XXS: case T_IQ2_XS: case T_IQ2_S: case T_IQ3_S:
        case T_BF16: case T_F32: return 16;
        default: return 32;
    }
}

// ---- reads -------------------------------------------------------------------------------------------
// Honeykrisp fetches 8 bytes for a 2-byte load and 4 for a 4-byte load: a narrow load reaches a quarter of the
// bandwidth (measured on the M1 Max: 70 GB/s for 2-byte loads against 290 GB/s for 4-byte and 16-byte loads -
// honeykrisp-strata's bench/bw_width).  The bulk of a block is therefore read with 4-byte loads.
//
// A 4-byte load may sit at any 2-byte aligned address (the blocks start at 2-byte aligned addresses and their
// sizes are not multiples of 4), so a 16-bit array is read two words at a time (rd2x2).  A field at an odd offset
// (the i-quant sign and scale bytes) is read a byte at a time: there are a few of them a block, and reading them
// as words costs more in arithmetic than it saves.
uint rd16(uint8_t* p) { return uint(*(uint16_t*)p); }
uint rd8(uint8_t* p) { return uint(*(uint8_t*)p); }
// the two 16-bit words of a 16-bit array at p[i] and p[i+1], read as one 4-byte load (the array is 2-byte aligned)
uint rd2x2(uint16_t* p, int i) {
    const uint16_t2 v = loadAligned<2>((uint16_t2*)(p + i));
    return uint(v.x) | (uint(v.y) << 16);
}
// the block's fp16 scale: one 2-byte read a block, so it stays narrow
float rdh(uint8_t* p) { return float(*(half*)p); }

// ---- threadgroup tables ---------------------------------------------------------------------------------------
// QUANT_CTAB (function constant 23): read the grids and the sign table from the constant arrays instead of a
// threadgroup copy; the same values.
[vk::constant_id(23)] const bool QUANT_CTAB = false;

groupshared uint t_signs[128];
groupshared uint t_iq3xxs[256];
groupshared uint t_iq3s[512];
groupshared uint t_iq2xxs[512];   // two words per grid entry (low, high)
groupshared uint t_iq2xs[1024];
groupshared uint t_iq2s[2048];
groupshared uint t_iq4[256];      // a byte -> (codebook[low nibble], codebook[high nibble]) as two halves

uint ksg(uint i) { return QUANT_CTAB ? ksigns_iq2xs[i] : t_signs[i]; }
uint g_iq3xxs(uint i) { return QUANT_CTAB ? iq3xxs_grid[i] : t_iq3xxs[i]; }
uint g_iq3s(uint i) { return QUANT_CTAB ? iq3s_grid[i] : t_iq3s[i]; }
uint g_iq2xxs(uint i, int w) { return QUANT_CTAB ? uint(iq2xxs_grid[i] >> (32 * w)) : t_iq2xxs[2 * i + w]; }
uint g_iq2xs(uint i, int w) { return QUANT_CTAB ? uint(iq2xs_grid[i] >> (32 * w)) : t_iq2xs[2 * i + w]; }
uint g_iq2s(uint i, int w) { return QUANT_CTAB ? uint(iq2s_grid[i] >> (32 * w)) : t_iq2s[2 * i + w]; }

void load_signs(uint tid, uint nth) {
    for (uint i = tid; i < 128; i += nth) t_signs[i] = ksigns_iq2xs[i];
}

uint iq4_pair(uint i) {
    return f32tof16(float(kvalues_iq4nl[i & 15])) | (f32tof16(float(kvalues_iq4nl[i >> 4])) << 16);
}

void load_tab<let F : int>(uint tid, uint nth) {
    if (F == T_IQ4_NL || F == T_IQ4_XS) {
        for (uint i = tid; i < 256; i += nth) t_iq4[i] = iq4_pair(i);
        return;
    }
    if (QUANT_CTAB) return;
    if (F == T_IQ3_XXS) {
        for (uint i = tid; i < 256; i += nth) t_iq3xxs[i] = iq3xxs_grid[i];
        load_signs(tid, nth);
    } else if (F == T_IQ3_S) {
        for (uint i = tid; i < 512; i += nth) t_iq3s[i] = iq3s_grid[i];
        load_signs(tid, nth);
    } else if (F == T_IQ2_XXS) {
        for (uint i = tid; i < 256; i += nth) { t_iq2xxs[2 * i] = uint(iq2xxs_grid[i]); t_iq2xxs[2 * i + 1] = uint(iq2xxs_grid[i] >> 32); }
        load_signs(tid, nth);
    } else if (F == T_IQ2_XS) {
        for (uint i = tid; i < 512; i += nth) { t_iq2xs[2 * i] = uint(iq2xs_grid[i]); t_iq2xs[2 * i + 1] = uint(iq2xs_grid[i] >> 32); }
        load_signs(tid, nth);
    } else if (F == T_IQ2_S) {
        for (uint i = tid; i < 1024; i += nth) { t_iq2s[2 * i] = uint(iq2s_grid[i]); t_iq2s[2 * i + 1] = uint(iq2s_grid[i] >> 32); }
        load_signs(tid, nth);
    }
}

// ---- unit decoders: return s, fill q[UNIT] (ui = unit index inside the block) ---------------------------------
void put_signed4(inout float q[MAXU], int o, uint g, uint s) {
    [ForceUnroll]
    for (int k = 0; k < 4; ++k) {
        const float v = float((g >> (8 * k)) & 255u);
        q[o + k] = (s & (1u << k)) != 0 ? -v : v;
    }
}

float iq4_lo(uint w) { return f16tof32(w & 0xFFFFu); }
float iq4_hi(uint w) { return f16tof32(w >> 16); }

// The unit decoders: return s, fill q[UNIT] (ui = unit index inside the block).
float deq<let F : int>(uint8_t* b, int ui, inout float q[MAXU]) {
    if (F == T_Q2_0) {   // 64 = 4 units of 16; code c (2 bits, LSB first) -> c - 1; scale d
        const uint16_t2 ac = loadAligned<2>((uint16_t2*)(b + 2 + 4 * ui));
        const uint a = uint(ac.x), c = uint(ac.y);
        [ForceUnroll]
        for (int j = 0; j < 8; ++j) {
            q[j] = float((a >> (2 * j)) & 3u) - 1.0f;
            q[8 + j] = float((c >> (2 * j)) & 3u) - 1.0f;
        }
        return rdh(b);
    } else if (F == T_IQ3_XXS) {   // 256 = 8 x 32; unit = half of a 32 (two grid pairs)
        const int ib = ui >> 1, h = ui & 1;
        uint16_t* ap = (uint16_t*)(b + 2 + 64 + 4 * ib);
        const uint aux = rd2x2(ap, 0);
        uint8_t* qp = b + 2 + 8 * ib + 4 * h;
        const uint qs = rd2x2((uint16_t*)qp, 0);
        [ForceUnroll]
        for (int l2 = 0; l2 < 2; ++l2) {
            const int l = 2 * h + l2;
            const uint s = ksg((aux >> (7 * l)) & 127u);
            put_signed4(q, 8 * l2, g_iq3xxs((qs >> (8 * 2 * l2)) & 255u), s);
            put_signed4(q, 8 * l2 + 4, g_iq3xxs((qs >> (8 * (2 * l2 + 1))) & 255u), s >> 4);
        }
        return rdh(b) * (0.5f + float(aux >> 28)) * 0.5f;
    } else if (F == T_IQ2_XXS) {   // per 32: two u32 (4 grid indices, then 4 x 7 sign bits + 4-bit scale)
        const int ib = ui >> 1, h = ui & 1;
        uint16_t* ap = (uint16_t*)(b + 2 + 8 * ib);
        const uint a0 = rd2x2(ap, 0), a1 = rd2x2(ap, 2);
        [ForceUnroll]
        for (int l2 = 0; l2 < 2; ++l2) {
            const int l = 2 * h + l2;
            const uint idx = (a0 >> (8 * l)) & 255u;
            const uint s = ksg((a1 >> (7 * l)) & 127u);
            put_signed4(q, 8 * l2, g_iq2xxs(idx, 0), s);
            put_signed4(q, 8 * l2 + 4, g_iq2xxs(idx, 1), s >> 4);
        }
        return rdh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
    } else if (F == T_IQ2_XS) {   // uint16 qs[32] (9-bit grid index + 7-bit sign index), scales[8] nibbles
        const int ib = ui >> 1, h = ui & 1;
        // the two grid indices of a unit are read together (four bytes)
        uint16_t* qp = (uint16_t*)(b + 2);
        const uint v2 = rd2x2(qp, 4 * ib + 2 * h);
        [ForceUnroll]
        for (int l2 = 0; l2 < 2; ++l2) {
            const uint v = (v2 >> (16 * l2)) & 0xFFFFu;
            const uint idx = v & 511u, s = ksg(v >> 9);
            put_signed4(q, 8 * l2, g_iq2xs(idx, 0), s);
            put_signed4(q, 8 * l2 + 4, g_iq2xs(idx, 1), s >> 4);
        }
        const uint sc = rd8(b + 66 + ib);
        return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15u)) * 0.25f;
    } else if (F == T_IQ2_S) {   // qs[32] low 8 bits of the 10-bit index, signs[32], qh[8], scales[8]
        const int ib = ui >> 1, h = ui & 1;
        const uint qh = rd8(b + 66 + ib);
        uint8_t* qp = b + 2 + 4 * ib;
        uint8_t* sp = b + 34 + 4 * ib;
        [ForceUnroll]
        for (int l2 = 0; l2 < 2; ++l2) {
            const int l = 2 * h + l2;
            const uint idx = uint(qp[l]) | ((qh << (8 - 2 * l)) & 0x300u);
            const uint s = uint(sp[l]);
            put_signed4(q, 8 * l2, g_iq2s(idx, 0), s);
            put_signed4(q, 8 * l2 + 4, g_iq2s(idx, 1), s >> 4);
        }
        const uint sc = rd8(b + 74 + ib);
        return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15u)) * 0.25f;
    } else if (F == T_IQ3_S) {   // qs[64] low 8 bits of 9-bit indices, qh[8], signs[32], scales[4]
        const int ib = ui >> 1, h = ui & 1;
        const uint qh = uint(b[66 + ib]);
        [ForceUnroll]
        for (int l2 = 0; l2 < 2; ++l2) {
            const int l = 2 * h + l2;
            const uint i1 = uint(b[2 + 8 * ib + 2 * l]) | ((qh << (8 - 2 * l)) & 256u);
            const uint i2 = uint(b[2 + 8 * ib + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256u);
            const uint s = uint(b[74 + 4 * ib + l]);
            put_signed4(q, 8 * l2, g_iq3s(i1), s);
            put_signed4(q, 8 * l2 + 4, g_iq3s(i2), s >> 4);
        }
        const uint sc = uint(b[106 + (ib >> 1)]);
        return rdh(b) * float(1 + 2 * int((sc >> (4 * (ib & 1))) & 15u));
    } else if (F == T_IQ4_NL || F == T_IQ4_XS) {   // low nibbles are elements 0..15, high 16..31
        uint16_t* p = (uint16_t*)(b + (F == T_IQ4_NL ? 2 : 8 + 16 * ui));
        [ForceUnroll]
        for (int j = 0; j < 8; j += 2) {
            const uint w = rd2x2(p, j);
            const uint v0 = w & 0xFFFFu, v1 = w >> 16;
            const uint w0 = t_iq4[v0 & 255u], w1 = t_iq4[v0 >> 8];
            const uint w2 = t_iq4[v1 & 255u], w3 = t_iq4[v1 >> 8];
            q[2 * j] = iq4_lo(w0);
            q[2 * j + 16] = iq4_hi(w0);
            q[2 * j + 1] = iq4_lo(w1);
            q[2 * j + 17] = iq4_hi(w1);
            q[2 * j + 2] = iq4_lo(w2);
            q[2 * j + 18] = iq4_hi(w2);
            q[2 * j + 3] = iq4_lo(w3);
            q[2 * j + 19] = iq4_hi(w3);
        }
        if (F == T_IQ4_NL) return rdh(b);
        const uint sh = rd16(b + 2);
        const int ls = int((rd8(b + 4 + (ui >> 1)) >> (4 * (ui & 1))) & 15u) | int(((sh >> (2 * ui)) & 3u) << 4);
        return rdh(b) * float(ls - 32);
    } else if (F == T_Q8_0) {
        uint16_t* p = (uint16_t*)(b + 2);
        [ForceUnroll]
        for (int j = 0; j < 16; j += 2) {
            const uint w = rd2x2(p, j);
            const uint v0 = w & 0xFFFFu, v1 = w >> 16;
            q[2 * j] = float(int(v0 << 24) >> 24);
            q[2 * j + 1] = float(int(v0 << 16) >> 24);
            q[2 * j + 2] = float(int(v1 << 24) >> 24);
            q[2 * j + 3] = float(int(v1 << 16) >> 24);
        }
        return rdh(b);
    } else if (F == T_Q4_0) {
        uint16_t* p = (uint16_t*)(b + 2);
        [ForceUnroll]
        for (int j = 0; j < 8; j += 2) {
            const uint w = rd2x2(p, j);
            const uint v0 = w & 0xFFFFu, v1 = w >> 16;
            q[2 * j] = float(v0 & 15u) - 8.0f;
            q[2 * j + 16] = float((v0 >> 4) & 15u) - 8.0f;
            q[2 * j + 1] = float((v0 >> 8) & 15u) - 8.0f;
            q[2 * j + 17] = float(v0 >> 12) - 8.0f;
            q[2 * j + 2] = float(v1 & 15u) - 8.0f;
            q[2 * j + 18] = float((v1 >> 4) & 15u) - 8.0f;
            q[2 * j + 3] = float((v1 >> 8) & 15u) - 8.0f;
            q[2 * j + 19] = float(v1 >> 12) - 8.0f;
        }
        return rdh(b);
    } else if (F == T_Q5_0 || F == T_Q5_1) {   // Q5_1: the 5-bit code (no offset; the min goes through the x sums)
        uint16_t* hp = (uint16_t*)(b + (F == T_Q5_0 ? 2 : 4));
        const uint hb = rd2x2(hp, 0);
        uint16_t* p = (uint16_t*)(b + (F == T_Q5_0 ? 6 : 8));
        const float off = F == T_Q5_0 ? 16.0f : 0.0f;
        [ForceUnroll]
        for (int j = 0; j < 8; j += 2) {
            const uint w = rd2x2(p, j);
            const uint v0 = w & 0xFFFFu, v1 = w >> 16;
            [ForceUnroll]
            for (int k = 0; k < 2; ++k) {
                const uint v = k == 0 ? v0 : v1;
                const int j0 = 2 * j + 2 * k, j1 = 2 * j + 2 * k + 1;
                q[j0] = float(int(v & 15u) | int(((hb >> j0) & 1u) << 4)) - off;
                q[j0 + 16] = float(int((v >> 4) & 15u) | int(((hb >> (j0 + 16)) & 1u) << 4)) - off;
                q[j1] = float(int((v >> 8) & 15u) | int(((hb >> j1) & 1u) << 4)) - off;
                q[j1 + 16] = float(int(v >> 12) | int(((hb >> (j1 + 16)) & 1u) << 4)) - off;
            }
        }
        return rdh(b);
    }
    return 0.0f;
}

// sum over groups of LPR consecutive lanes (LPR = 1, 2, 4, 8, 16 or 32); the result is valid in every lane
float lane_sum<let LPR : int>(float v) {
    if (LPR == 32) return WaveActiveSum(v);
    const uint lane = WaveGetLaneIndex();
    [ForceUnroll]
    for (uint o = LPR / 2; o > 0; o >>= 1) v += WaveReadLaneAt(v, lane ^ o);
    return v;
}
