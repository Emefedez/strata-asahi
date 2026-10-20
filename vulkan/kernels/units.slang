// vulkan/kernels/units.slang - metal/kernels/units.metal in Slang: the dot of one unit with the window's tokens.
//
// X_load<F>(xrow, s16, u, xs, xsum) prepares unit u's x slice of one token, once per unit and shared by every
// weight row the lane reads; where the codes are read with the mask trick (an AND keeps a field in place, so the code
// arrives multiplied by 2^k), the x values are pre-scaled by 2^-k by xprep: exact in float.

void load16(float* p, inout float xs[MAXU], int o) {
    [ForceUnroll]
    for (int j = 0; j < 4; ++j) {
        const float4 a = loadAligned<16>((float4*)(p + 4 * j));
        xs[o + 4 * j] = a.x;
        xs[o + 4 * j + 1] = a.y;
        xs[o + 4 * j + 2] = a.z;
        xs[o + 4 * j + 3] = a.w;
    }
}

// ---- prepared x (xprep): XP holds, per token row of n floats, the kinds
//   XK_RAW  x                       (every format without the mask trick)
//   XK_Q2   x * 4^-(i % 8)          (Q2_0: 2-bit fields in place in a 16-bit word)
//   XK_P4   x * 2^-(8 (i % 4))      (Q4_K, Q5_K: byte fields in place in a 32-bit word)
//   XK_P2   x * 2^-(8 (i % 2))      (Q6_K, Q3_K: byte fields in place in a 16-bit word)
// and S16 = the sum of every 16 consecutive raw x (the formats with an offset or a min).
#define XK_RAW 0
#define XK_Q2 1
#define XK_P4 2
#define XK_P2 3
int x_kind(int F) {
    switch (F) {
        case T_Q2_0: return XK_Q2;
        case T_Q4_K: case T_Q5_K: return XK_P4;
        case T_Q6_K: case T_Q3_K: return XK_P2;
        default: return XK_RAW;
    }
}

void X_load<let F : int>(float* xp, float* s16, int u, inout float xs[MAXU], inout float xsum[2]) {
    if (F == T_Q2_0) {
        load16(xp + 16 * u, xs, 0);
        xsum[0] = s16[u];
        xsum[1] = 0.0f;
    } else if (F == T_Q4_K || F == T_Q5_K) {   // unit (block b, chunk c, half m): run0 at 256b + 64c + 16m, run1 = run0 + 32
        const int p0 = 256 * (u >> 3) + 64 * ((u >> 1) & 3) + 16 * (u & 1);
        load16(xp + p0, xs, 0);
        load16(xp + p0 + 32, xs, 16);
        xsum[0] = s16[p0 >> 4];
        xsum[1] = s16[(p0 >> 4) + 2];
    } else if (F == T_Q6_K) {   // unit (block b, half n, m, p): run0 at 256b + 128n + 32p + 16m, run1 = run0 + 64
        const int p0 = 256 * (u >> 3) + 128 * ((u >> 2) & 1) + 32 * (u & 1) + 16 * ((u >> 1) & 1);
        load16(xp + p0, xs, 0);
        load16(xp + p0 + 64, xs, 16);
        xsum[0] = s16[p0 >> 4];
        xsum[1] = s16[(p0 >> 4) + 4];
    } else if (F == T_Q5_1) {   // 32 contiguous raw x, the two per-16 sums formed from them
        load16(xp + 32 * u, xs, 0);
        load16(xp + 32 * u + 16, xs, 16);
        float a = 0.0f, b = 0.0f;
        [ForceUnroll]
        for (int j = 0; j < 16; ++j) { a += xs[j]; b += xs[16 + j]; }
        xsum[0] = a;
        xsum[1] = b;
    } else if (F == T_Q3_K) {   // 32 contiguous, sums per 16
        load16(xp + 32 * u, xs, 0);
        load16(xp + 32 * u + 16, xs, 16);
        xsum[0] = s16[2 * u];
        xsum[1] = s16[2 * u + 1];
    } else {
        const int UNIT = f_UNIT(F);
        [unroll]   // the bound is a constant once f_UNIT(F) is folded (the driver unrolls it)
        for (int k = 0; k < UNIT; k += 16) load16(xp + u * UNIT + k, xs, k);
        xsum[0] = 0.0f;
        xsum[1] = 0.0f;
    }
}

// xprep: one thread per 16 consecutive values of one token row; XP = [4 kinds][n_tok][n] then S16 [n_tok][n / 16]
struct XprepPush {
    float* x;
    float* xp;
    uint n;
    uint n_tok;
};

[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void xprep(uint3 gidv : SV_DispatchThreadID, uniform XprepPush a) {
    const uint gid = gidv.x;
    const uint per = a.n / 16;
    if (gid >= per * a.n_tok) return;
    const uint t = gid / per, i = gid % per;
    float* src = a.x + t * a.n + 16 * i;
    const uint plane = a.n_tok * a.n;
    float* raw = a.xp + t * a.n + 16 * i;
    float v[MAXU];
    load16(src, v, 0);
    float s0 = 0, s1 = 0;
    [ForceUnroll]
    for (int j = 0; j < 16; j += 2) { s0 += v[j]; s1 += v[j + 1]; }
    const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
    const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
    [ForceUnroll]
    for (int j = 0; j < 16; ++j) {
        raw[j] = v[j];
        raw[plane + j] = v[j] * q2s[j & 7];
        raw[2 * plane + j] = v[j] * p4s[j & 3];
        raw[3 * plane + j] = (j & 1) != 0 ? v[j] * 0.00390625f : v[j];
    }
    a.xp[4 * plane + t * per + i] = s0 + s1;
}

// ---- decoders: Dec_run<F>(row, u, q, sc) decodes unit u of a row into q[UNIT] and sc = (d1, d2, m1, m2), so that
// for any token: dot = d1 * sum_{j < UNIT/2} q[j] xs[j] + d2 * sum_{j >= UNIT/2} q[j] xs[j] - (m1 xsum0 + m2 xsum1).
void scale_min_k4(int j, uint8_t* q, out float d, out float m) {
    if (j < 4) {
        d = float(uint(q[j]) & 63u);
        m = float(uint(q[j + 4]) & 63u);
    } else {
        d = float((uint(q[j + 4]) & 0xFu) | ((uint(q[j - 4]) >> 6) << 4));
        m = float((uint(q[j + 4]) >> 4) | ((uint(q[j]) >> 6) << 4));
    }
}
void bytes4(uint w, inout float q[MAXU], int o) {
    q[o] = float(w & 0xFFu);
    q[o + 1] = float(w & 0xFF00u);
    q[o + 2] = float(w & 0xFF0000u);
    q[o + 3] = float(w & 0xFF000000u);
}
void bytes2(uint w, inout float q[MAXU], int o) {
    q[o] = float(w & 0xFFu);
    q[o + 1] = float(w & 0xFF00u);
}

void Dec_run<let F : int>(uint8_t* row, int u, inout float q[MAXU], out float4 sc) {
    if (F == T_Q2_0) {
        uint8_t* b = row + (u >> 2) * 18;
        // the unit's two halfwords as one 2 x 16-bit load (2-byte aligned): Mesa issues two narrow loads otherwise
        const uint16_t2 ac = loadAligned<2>((uint16_t2*)(b + 2 + 4 * (u & 3)));
        const uint a = uint(ac.x), c = uint(ac.y);
        [ForceUnroll]
        for (int j = 0; j < 8; ++j) {
            q[j] = float(a & (3u << (2 * j)));
            q[8 + j] = float(c & (3u << (2 * j)));
        }
        const float d = rdh(b);
        sc = float4(d, d, d, 0.0f);
    } else if (F == T_Q2_0R) {   // Q2_0 with raw x: q = c - 1, one scale
        uint8_t* b = row + (u >> 2) * 18;
        uint16_t* p = (uint16_t*)(b + 2 + 4 * (u & 3));
        const uint a = uint(p[0]), c = uint(p[1]);
        [ForceUnroll]
        for (int j = 0; j < 8; ++j) {
            q[j] = float((a >> (2 * j)) & 3u) - 1.0f;
            q[8 + j] = float((c >> (2 * j)) & 3u) - 1.0f;
        }
        const float d = rdh(b);
        sc = float4(d, d, 0.0f, 0.0f);
    } else if (F == T_Q5_1) {   // w = d q + m:  sum = d (q . x) + m (sum x)
        uint8_t* b = row + u * 24;
        const float d = deq<T_Q5_1>(b, 0, q), m = rdh(b + 2);
        sc = float4(d, d, -m, -m);
    } else if (F == T_Q4_K || F == T_Q5_K) {
        const int BBk = F == T_Q4_K ? 144 : 176;
        uint8_t* b = row + (u >> 3) * BBk;
        const int c = (u >> 1) & 3, m = u & 1;
        const float d = rdh(b), dm = rdh(b + 2);
        float s1, m1, s2, m2;
        scale_min_k4(2 * c, b + 4, s1, m1);
        scale_min_k4(2 * c + 1, b + 4, s2, m2);
        uint* qp = (uint*)(b + (F == T_Q4_K ? 16 : 48) + 32 * c + 16 * m);
        uint* hp = (uint*)(b + 16 + 16 * m);
        [ForceUnroll]
        for (int w = 0; w < 4; ++w) {
            const uint v = qp[w];
            if (F == T_Q4_K) {
                bytes4(v & 0x0F0F0F0Fu, q, 4 * w);
                bytes4((v >> 4) & 0x0F0F0F0Fu, q, 16 + 4 * w);
            } else {
                const uint h = hp[w];
                bytes4((v & 0x0F0F0F0Fu) | (((h >> (2 * c)) & 0x01010101u) << 4), q, 4 * w);
                bytes4(((v >> 4) & 0x0F0F0F0Fu) | (((h >> (2 * c + 1)) & 0x01010101u) << 4), q, 16 + 4 * w);
            }
        }
        sc = float4(d * s1, d * s2, dm * m1, dm * m2);
    } else if (F == T_Q6_K) {
        uint8_t* b = row + (u >> 3) * 210;
        const int n = (u >> 2) & 1, m = (u >> 1) & 1, p = u & 1;
        uint16_t* ql = (uint16_t*)(b + 64 * n + 32 * p + 16 * m);
        uint16_t* qh = (uint16_t*)(b + 128 + 32 * n + 16 * m);
        int8_t* scs = (int8_t*)(b + 192);
        const float d = rdh(b + 208);
        [ForceUnroll]
        for (int w = 0; w < 8; ++w) {
            const uint l = uint(ql[w]), h = uint(qh[w]);
            bytes2((l & 0x0F0Fu) | (((h >> (2 * p)) & 0x0303u) << 4), q, 2 * w);
            bytes2(((l >> 4) & 0x0F0Fu) | (((h >> (2 * p + 4)) & 0x0303u) << 4), q, 16 + 2 * w);
        }
        const float d1 = d * float(scs[8 * n + m + 2 * p]), d2 = d * float(scs[8 * n + m + 2 * p + 4]);
        sc = float4(d1, d2, 32.0f * d1, 32.0f * d2);
    } else if (F == T_Q3_K) {
        uint8_t* b = row + (u >> 3) * 110;
        const int n = (u >> 2) & 1, j = u & 3;
        uint16_t* qv = (uint16_t*)(b + 32 + 32 * n);   // odd blocks are only 2-aligned: ushort reads
        uint16_t* hv = (uint16_t*)b;
        uint8_t* s = b + 96;
        const int i1 = 8 * n + 2 * j, i2 = i1 + 1;
        const int l1 = int(i1 < 8 ? uint(s[i1]) & 0xFu : uint(s[i1 - 8]) >> 4) | int(((uint(s[8 + (i1 & 3)]) >> (2 * (i1 >> 2))) & 3u) << 4);
        const int l2 = int(i2 < 8 ? uint(s[i2]) & 0xFu : uint(s[i2 - 8]) >> 4) | int(((uint(s[8 + (i2 & 3)]) >> (2 * (i2 >> 2))) & 3u) << 4);
        const float d = rdh(b + 108);
        const int hs = 4 * n + j;
        [ForceUnroll]
        for (int w = 0; w < 8; ++w) {
            bytes2(((uint(qv[w]) >> (2 * j)) & 0x0303u) | (((uint(hv[w]) >> hs) & 0x0101u) << 2), q, 2 * w);
            bytes2(((uint(qv[w + 8]) >> (2 * j)) & 0x0303u) | (((uint(hv[w + 8]) >> hs) & 0x0101u) << 2), q, 16 + 2 * w);
        }
        const float d1 = d * float(l1 - 32), d2 = d * float(l2 - 32);
        sc = float4(d1, d2, 4.0f * d1, 4.0f * d2);
    } else if (F == T_F32) {   // 16 floats a unit
        float* p = (float*)(row + 64 * u);
        [ForceUnroll]
        for (int k = 0; k < 4; ++k) {
            const float4 v = loadAligned<16>((float4*)(p + 4 * k));
            q[4 * k] = v.x; q[4 * k + 1] = v.y; q[4 * k + 2] = v.z; q[4 * k + 3] = v.w;
        }
        sc = float4(1.0f, 1.0f, 0.0f, 0.0f);
    } else if (F == T_BF16) {
        uint4* p = (uint4*)(row + 32 * u);
        const uint4 a = loadAligned<16>(p), c = loadAligned<16>(p + 1);
        const uint v[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
        [ForceUnroll]
        for (int k = 0; k < 8; ++k) {
            q[2 * k] = asfloat(v[k] << 16);
            q[2 * k + 1] = asfloat(v[k] & 0xFFFF0000u);
        }
        sc = float4(1.0f, 1.0f, 0.0f, 0.0f);
    } else {
        const int UNIT = f_UNIT(F), UPB = f_QK(F) / UNIT;
        const float s = deq<F>(row + (u / UPB) * f_BB(F), u % UPB, q);
        sc = float4(s, s, 0.0f, 0.0f);
    }
}

// acc += unit dot of decoded q / sc with one token's prepared slice
float unit_combine(int UNIT, float q[MAXU], float4 sc, float xs[MAXU], float xsum[2]) {
    const int H = UNIT / 2;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    [unroll]
    for (int j = 0; j < H; j += 2) {
        s0 = fma(q[j], xs[j], s0);
        s1 = fma(q[j + 1], xs[j + 1], s1);
        s2 = fma(q[H + j], xs[H + j], s2);
        s3 = fma(q[H + j + 1], xs[H + j + 1], s3);
    }
    return sc.x * (s0 + s1) + sc.y * (s2 + s3) - (sc.z * xsum[0] + sc.w * xsum[1]);
}
