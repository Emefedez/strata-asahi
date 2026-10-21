// vulkan/kernels/gdn_prompt.slang - metal/kernels/gdn_prompt.metal in Slang (after gdn.slang): a GDN layer's
// recurrence for a PROMPT chunk of T tokens, in three passes (see the MSL for the algebra):
//   gdn_pre    per (token, head): conv -> SiLU; q, k L2 norm * 1/sqrt(128); kq = k . q; gate, beta; hist_out
//   gdn_scan   per (value head, columns): the state in registers, a token at a time (gdn_scan_<CB>: a lane owns a
//              whole column; gdn_scanr_<IB>: a column's rows split over IB simdgroups; gdn_scan_v: scanr 8 in float4s)
//   gdn_post   per (token, head): y = o * rsqrt(mean(o^2) + eps) * gamma * sigmoid(z) -> fp16 (and fp32)
//   gdn_cprep / gdn_cstate / gdn_cout: the chunked (WY) form, 32-token chunks
// State layout as Strata's: S[(i * h_v + h) * 128 + j].  Value head h reads q/k head h % h_k.

struct GdnPArgs {
    uint n_tok;
    uint h_k;
    uint h_v;
    uint channels;
    float eps;
    uint f32;
};

// a probe for the bench (function constant 6): 1 = no per-token exchange (wrong results), 2 = no state update
[vk::constant_id(6)] const int GDN_PROBE = 0;

float4 gp_ld4(float* p) { return loadAligned<16>((float4*)p); }

float gp_conv(float* qkv, float* hist, float* cw, int ch, int t, int channels) {
    float s = 0.0f;
    [ForceUnroll]
    for (int k = 0; k < 4; ++k) {
        const int src = t - 3 + k;
        const float x = src >= 0 ? qkv[src * channels + ch] : hist[ch * 3 + (src + 3)];
        s = fma(x, cw[ch * 4 + k], s);
    }
    s = s + 0.0f;
    return s / (1.0f + exp(-s));
}

struct GdnPrePush {
    float* qkv;
    float* hist;
    float* conv_w;
    float* alpha;
    float* beta;
    float* dt;
    float* ssm_a;
    float* qn;
    float* kn;
    float* vc;
    float2* gb;
    float* kq;
    float* hist_out;
    GdnPArgs a;
};

groupshared float g_gp_part[8];
groupshared float g_gp_prod[128];

// grid (h_k + h_v / 2, T), 256 threads.  x < h_k: q (threads 0..127) and k (128..255) of key head x; else value heads
// 2 (x - h_k) and + 1.
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gdn_pre(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform GdnPrePush pc) {
    const int S = 128;
    const GdnPArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int x = int(tg.x), t = int(tg.y), hk_n = int(a.h_k), hv = int(a.h_v), C = int(a.channels), T = int(a.n_tok);
    int ch;
    if (x < hk_n) ch = tid < 128 ? x * S + tid : hk_n * S + x * S + (tid - 128);
    else ch = 2 * hk_n * S + (2 * (x - hk_n) + (tid >> 7)) * S + (tid & 127);
    const float cv = gp_conv(pc.qkv, pc.hist, pc.conv_w, ch, t, C);
    if (t == T - 1)
        for (int k = 0; k < 3; ++k) {
            const int src = T - 3 + k;
            pc.hist_out[ch * 3 + k] = src >= 0 ? pc.qkv[src * C + ch] : pc.hist[ch * 3 + (src + 3)];
        }
    if (x < hk_n) {
        const float ssq = WaveActiveSum(cv * cv);
        if (lane == 0) g_gp_part[sg] = ssq;
        GroupMemoryBarrierWithGroupSync();
        const int base = tid < 128 ? 0 : 4;
        const float tot = (g_gp_part[base] + g_gp_part[base + 1]) + (g_gp_part[base + 2] + g_gp_part[base + 3]);
        const float rs = rsqrt(tot / float(S) + a.eps / float(S));
        const float nv = (rs * cv) * (1.0f / sqrt(float(S)));
        if (tid < 128) {
            pc.qn[(t * hk_n + x) * S + tid] = nv;
            g_gp_prod[tid] = nv;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid >= 128) {
            pc.kn[(t * hk_n + x) * S + tid - 128] = nv;
            const float p = WaveActiveSum(nv * g_gp_prod[tid - 128]);
            if (lane == 0) g_gp_part[sg] = p;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0) pc.kq[t * hk_n + x] = (g_gp_part[4] + g_gp_part[5]) + (g_gp_part[6] + g_gp_part[7]);
    } else {
        const int h = 2 * (x - hk_n) + (tid >> 7);
        pc.vc[(t * hv + h) * S + (tid & 127)] = cv;
        if ((tid & 127) == 0) {
            const float v = pc.alpha[t * hv + h] + pc.dt[h];
            const float e = exp(v), u = 1.0f + e;
            const float sp = v > 20.0f ? v : (u == 1.0f ? e : log(u) * e / (u - 1.0f));
            pc.gb[t * hv + h] = float2(exp(sp * pc.ssm_a[h]), 1.0f / (1.0f + exp(-pc.beta[t * hv + h])));
        }
    }
}

struct GdnScanPush {
    float* state;
    float* qn;
    float* kn;
    float* vc;
    float2* gb;
    float* kq;
    float* o;
    GdnPArgs a;
};

// TB = 16 tokens of q and k staged at a time (float4 x 32 = one head's 128 values)
groupshared float4 g_gs_q[16 * 32];
groupshared float4 g_gs_k[16 * 32];

// grid (h_v * 4 / CB), CB * 32 threads: a lane owns a whole column of the head's state
void gdn_scan_t<let CB : int>(GdnScanPush pc, uint3 tg, uint tidu) {
    const int S = 128, NB = 4 / CB, TB = 16;
    const GdnPArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int h = int(tg.x) / NB, hk = h % int(a.h_k), hv = int(a.h_v), hkn = int(a.h_k);
    const int j = ((int(tg.x) % NB) * CB + sg) * 32 + lane;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    float st[128];
    [ForceUnroll] for (int i = 0; i < S; ++i) st[i] = pc.state[(i * hv + h) * S + j];
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        GroupMemoryBarrierWithGroupSync();
        for (int idx = tid; idx < nb * 32; idx += CB * 32) {
            const int tt = idx >> 5, w = idx & 31;
            g_gs_q[tt * 32 + w] = gp_ld4(pc.qn + (((t0 + tt) * hkn + hk) * 32 + w) * 4);
            g_gs_k[tt * 32 + w] = gp_ld4(pc.kn + (((t0 + tt) * hkn + hk) * 32 + w) * 4);
        }
        GroupMemoryBarrierWithGroupSync();
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt;
            const float2 g_b = pc.gb[t * hv + h];
            const float kqv = pc.kq[t * hkn + hk];
            const float v = pc.vc[(t * hv + h) * S + j];
            float ka = 0.0f, kb = 0.0f, qa = 0.0f, qb = 0.0f;
            [ForceUnroll]
            for (int w = 0; w < 32; w += 2) {
                const float4 k0 = g_gs_k[tt * 32 + w], k1 = g_gs_k[tt * 32 + w + 1];
                const float4 q0 = g_gs_q[tt * 32 + w], q1 = g_gs_q[tt * 32 + w + 1];
                ka = fma(st[4 * w], k0.x, ka); ka = fma(st[4 * w + 1], k0.y, ka);
                ka = fma(st[4 * w + 2], k0.z, ka); ka = fma(st[4 * w + 3], k0.w, ka);
                kb = fma(st[4 * w + 4], k1.x, kb); kb = fma(st[4 * w + 5], k1.y, kb);
                kb = fma(st[4 * w + 6], k1.z, kb); kb = fma(st[4 * w + 7], k1.w, kb);
                qa = fma(st[4 * w], q0.x, qa); qa = fma(st[4 * w + 1], q0.y, qa);
                qa = fma(st[4 * w + 2], q0.z, qa); qa = fma(st[4 * w + 3], q0.w, qa);
                qb = fma(st[4 * w + 4], q1.x, qb); qb = fma(st[4 * w + 5], q1.y, qb);
                qb = fma(st[4 * w + 6], q1.z, qb); qb = fma(st[4 * w + 7], q1.w, qb);
            }
            const float g = g_b.x, kv = ka + kb, qv = qa + qb;
            const float delta = (v - g * kv) * g_b.y;
            pc.o[(t * hv + h) * S + j] = fma(g, qv, delta * kqv) * scale;
            [ForceUnroll]
            for (int w = 0; w < 32; ++w) {
                const float4 k4 = g_gs_k[tt * 32 + w];
                st[4 * w] = fma(g, st[4 * w], k4.x * delta);
                st[4 * w + 1] = fma(g, st[4 * w + 1], k4.y * delta);
                st[4 * w + 2] = fma(g, st[4 * w + 2], k4.z * delta);
                st[4 * w + 3] = fma(g, st[4 * w + 3], k4.w * delta);
            }
        }
    }
    [ForceUnroll] for (int i = 0; i < S; ++i) pc.state[(i * hv + h) * S + j] = st[i];
}

groupshared float2 g_gs_red[2 * 8 * 32];
groupshared float g_gs_v[16 * 32];
groupshared float4 g_gs_gk[16];

// stage TB tokens of q, k, the head's 32 v columns and (gate, beta, kq)
void gp_stage_r(GdnScanPush pc, int t0, int nb, int hk, int h, int j, int lane, int tid, int nthreads, int TB) {
    const int S = 128, hv = int(pc.a.h_v), hkn = int(pc.a.h_k);
    for (int idx = tid; idx < nb * 32; idx += nthreads) {
        const int tt = idx >> 5, w = idx & 31;
        g_gs_q[tt * 32 + w] = gp_ld4(pc.qn + (((t0 + tt) * hkn + hk) * 32 + w) * 4);
        g_gs_k[tt * 32 + w] = gp_ld4(pc.kn + (((t0 + tt) * hkn + hk) * 32 + w) * 4);
        g_gs_v[tt * 32 + w] = pc.vc[((t0 + tt) * hv + h) * S + (j - lane) + w];
        if (w == 0) {
            const float2 g_b = pc.gb[(t0 + tt) * hv + h];
            g_gs_gk[tt] = float4(g_b.x, g_b.y, pc.kq[(t0 + tt) * hkn + hk], 0.0f);
        }
    }
}

// the rows of a column split over IB simdgroups: grid (h_v * 4), IB * 32 threads; lane = column; per token the IB
// partial dots are exchanged through threadgroup memory (double-buffered: one barrier a token)
void gdn_scan_r<let IB : int, let TB : int>(GdnScanPush pc, uint3 tg, uint tidu) {
    const int S = 128, R = S / IB, R4 = R / 4;
    const GdnPArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int h = int(tg.x) / 4, hk = h % int(a.h_k), hv = int(a.h_v);
    const int j = (int(tg.x) % 4) * 32 + lane, i0 = sg * R;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    float st[R];
    [ForceUnroll] for (int i = 0; i < R; ++i) st[i] = pc.state[((i0 + i) * hv + h) * S + j];
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        GroupMemoryBarrierWithGroupSync();
        gp_stage_r(pc, t0, nb, hk, h, j, lane, tid, IB * 32, TB);
        GroupMemoryBarrierWithGroupSync();
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt, slot = t & 1;
            float ka = 0.0f, qa = 0.0f;
            [ForceUnroll]
            for (int w = 0; w < R4; ++w) {
                const float4 k4 = g_gs_k[tt * 32 + i0 / 4 + w], q4 = g_gs_q[tt * 32 + i0 / 4 + w];
                ka = fma(st[4 * w], k4.x, ka); ka = fma(st[4 * w + 1], k4.y, ka);
                ka = fma(st[4 * w + 2], k4.z, ka); ka = fma(st[4 * w + 3], k4.w, ka);
                qa = fma(st[4 * w], q4.x, qa); qa = fma(st[4 * w + 1], q4.y, qa);
                qa = fma(st[4 * w + 2], q4.z, qa); qa = fma(st[4 * w + 3], q4.w, qa);
            }
            const float4 g_b = g_gs_gk[tt];
            const float kqv = g_b.z;
            const float v = g_gs_v[tt * 32 + lane];
            float2 s = float2(ka, qa);
            if (GDN_PROBE != 1) {
                g_gs_red[(slot * 8 + sg) * 32 + lane] = s;
                GroupMemoryBarrierWithGroupSync();
                s = g_gs_red[(slot * 8 + 0) * 32 + lane];
                [ForceUnroll] for (int b = 1; b < IB; ++b) s += g_gs_red[(slot * 8 + b) * 32 + lane];
            }
            const float g = g_b.x;
            const float delta = (v - g * s.x) * g_b.y;
            if (sg == 0) pc.o[(t * hv + h) * S + j] = fma(g, s.y, delta * kqv) * scale;
            if (GDN_PROBE != 2) {
                [ForceUnroll]
                for (int w = 0; w < R4; ++w) {
                    const float4 k4 = g_gs_k[tt * 32 + i0 / 4 + w];
                    st[4 * w] = fma(g, st[4 * w], k4.x * delta);
                    st[4 * w + 1] = fma(g, st[4 * w + 1], k4.y * delta);
                    st[4 * w + 2] = fma(g, st[4 * w + 2], k4.z * delta);
                    st[4 * w + 3] = fma(g, st[4 * w + 3], k4.w * delta);
                }
            }
        }
    }
    [ForceUnroll] for (int i = 0; i < R; ++i) pc.state[((i0 + i) * hv + h) * S + j] = st[i];
}

// gdn_scan_r<8, 16> with the 16 state rows of a lane in four float4 variables (no array)
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gdn_scan_v(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform GdnScanPush pc) {
    const int S = 128, IB = 8, TB = 16;
    const GdnPArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int h = int(tg.x) / 4, hk = h % int(a.h_k), hv = int(a.h_v);
    const int j = (int(tg.x) % 4) * 32 + lane, i0 = sg * 16, w0 = sg * 4;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    float* sp = pc.state + (i0 * hv + h) * S + j;
    const int rs = hv * S;
    float4 s0 = float4(sp[0], sp[rs], sp[2 * rs], sp[3 * rs]);
    float4 s1 = float4(sp[4 * rs], sp[5 * rs], sp[6 * rs], sp[7 * rs]);
    float4 s2 = float4(sp[8 * rs], sp[9 * rs], sp[10 * rs], sp[11 * rs]);
    float4 s3 = float4(sp[12 * rs], sp[13 * rs], sp[14 * rs], sp[15 * rs]);
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        GroupMemoryBarrierWithGroupSync();
        gp_stage_r(pc, t0, nb, hk, h, j, lane, tid, IB * 32, TB);
        GroupMemoryBarrierWithGroupSync();
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt, slot = t & 1;
            const float4 k0 = g_gs_k[tt * 32 + w0], k1 = g_gs_k[tt * 32 + w0 + 1], k2 = g_gs_k[tt * 32 + w0 + 2], k3 = g_gs_k[tt * 32 + w0 + 3];
            const float4 q0 = g_gs_q[tt * 32 + w0], q1 = g_gs_q[tt * 32 + w0 + 1], q2 = g_gs_q[tt * 32 + w0 + 2], q3 = g_gs_q[tt * 32 + w0 + 3];
            const float ka = (dot(s0, k0) + dot(s1, k1)) + (dot(s2, k2) + dot(s3, k3));
            const float qa = (dot(s0, q0) + dot(s1, q1)) + (dot(s2, q2) + dot(s3, q3));
            g_gs_red[(slot * 8 + sg) * 32 + lane] = float2(ka, qa);
            const float4 g_b = g_gs_gk[tt];
            const float v = g_gs_v[tt * 32 + lane];
            GroupMemoryBarrierWithGroupSync();
            float2 r = g_gs_red[(slot * 8) * 32 + lane];
            for (int b = 1; b < IB; ++b) r += g_gs_red[(slot * 8 + b) * 32 + lane];
            const float g = g_b.x;
            const float delta = (v - g * r.x) * g_b.y;
            if (sg == 0) pc.o[(t * hv + h) * S + j] = fma(g, r.y, delta * g_b.z) * scale;
            s0 = fma(float4(g), s0, k0 * delta);
            s1 = fma(float4(g), s1, k1 * delta);
            s2 = fma(float4(g), s2, k2 * delta);
            s3 = fma(float4(g), s3, k3 * delta);
        }
    }
    sp[0] = s0.x; sp[rs] = s0.y; sp[2 * rs] = s0.z; sp[3 * rs] = s0.w;
    sp[4 * rs] = s1.x; sp[5 * rs] = s1.y; sp[6 * rs] = s1.z; sp[7 * rs] = s1.w;
    sp[8 * rs] = s2.x; sp[9 * rs] = s2.y; sp[10 * rs] = s2.z; sp[11 * rs] = s2.w;
    sp[12 * rs] = s3.x; sp[13 * rs] = s3.y; sp[14 * rs] = s3.z; sp[15 * rs] = s3.w;
}

struct GdnPostPush {
    float* o;
    float* z;
    float* gamma;
    half* yh;
    float* y;
    GdnPArgs a;
};

groupshared float g_gpo_part[4];

// grid (h_v, T), 128 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gdn_post(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GdnPostPush pc) {
    const GdnPArgs a = pc.a;
    const int h = int(tg.x), t = int(tg.y), hv = int(a.h_v);
    const uint i = uint(t * hv + h) * 128 + tid;
    const float ov = pc.o[i];
    const float ss = WaveActiveSum(ov * ov);
    if (WaveGetLaneIndex() == 0) g_gpo_part[tid / 32] = ss;
    GroupMemoryBarrierWithGroupSync();
    const float tot = (g_gpo_part[0] + g_gpo_part[1]) + (g_gpo_part[2] + g_gpo_part[3]);
    const float rs = rsqrt(tot / 128.0f + a.eps);
    const float yv = ((rs * ov) * pc.gamma[tid]) * (1.0f / (1.0f + exp(-pc.z[i])));
    pc.yh[i] = half(yv);
    if (a.f32 != 0) pc.y[i] = yv;
}

#define SCAN(CB) [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)] \
    void gdn_scan_##CB(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GdnScanPush pc) { gdn_scan_t<CB>(pc, tg, tid); }
SCAN(1) SCAN(2) SCAN(4)
#define SCANR(IB, TB, NAME) [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)] \
    void NAME(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GdnScanPush pc) { gdn_scan_r<IB, TB>(pc, tg, tid); }
SCANR(2, 16, gdn_scanr_2) SCANR(4, 16, gdn_scanr_4) SCANR(8, 16, gdn_scanr_8) SCANR(4, 8, gdn_scanr_4_8)

// ---- the chunked form (chunks of GC = 32 tokens) ------------------------------------------------------------------
static const int GC = 32;

struct GdnCPrepPush {
    float* qn;
    float* kn;
    float* vc;
    float2* gb;
    float* U;
    float* W;
    float* lgo;
    GdnPArgs a;
};

groupshared float g_cp_A[32 * 32];
groupshared float g_cp_lg[32];
groupshared float g_cp_bt[32];

// grid (h_v, chunks), 256 threads.  U, W: [h][chunk][GC][128]; lgo: [h][chunk][GC]
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gdn_cprep(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform GdnCPrepPush pc) {
    const int S = 128;
    const GdnPArgs a = pc.a;
    const int tid = int(tidu);
    const int h = int(tg.x), c = int(tg.y), hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC, t0 = c * GC, nb = min(GC, T - t0);
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < GC; ++t) {
            if (t < nb) s += max(log(pc.gb[(t0 + t) * hv + h].x), -80.0f);
            g_cp_lg[t] = s;
            g_cp_bt[t] = t < nb ? pc.gb[(t0 + t) * hv + h].y : 0.0f;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (int idx = tid; idx < GC * GC; idx += 256) {
        const int t = idx / GC, s = idx % GC;
        float v = 0.0f;
        if (s < t && t < nb) {
            float* kt = pc.kn + ((t0 + t) * hkn + hk) * S;
            float* ks = pc.kn + ((t0 + s) * hkn + hk) * S;
            float d = 0.0f;
            for (int i = 0; i < S / 4; ++i) d += dot(gp_ld4(kt + 4 * i), gp_ld4(ks + 4 * i));
            v = g_cp_bt[t] * d * exp(g_cp_lg[t] - g_cp_lg[s]);
        }
        g_cp_A[t * GC + s] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    const int j = tid & 127;
    const bool isw = tid >= 128;
    float x[32];
    [ForceUnroll]
    for (int t = 0; t < GC; ++t) {
        float b = 0.0f;
        if (t < nb) b = isw ? g_cp_bt[t] * exp(g_cp_lg[t]) * pc.kn[((t0 + t) * hkn + hk) * S + j] : g_cp_bt[t] * pc.vc[((t0 + t) * hv + h) * S + j];
        [ForceUnroll] for (int s = 0; s < t; ++s) b = fma(-g_cp_A[t * GC + s], x[s], b);
        x[t] = b;
    }
    float* o = (isw ? pc.W : pc.U) + (uint64_t(h) * uint(NC) + uint(c)) * uint(GC * S) + uint(j);
    [ForceUnroll] for (int t = 0; t < GC; ++t) o[t * S] = x[t];
    if (tid < GC) pc.lgo[(h * NC + c) * GC + tid] = g_cp_lg[tid];
}

struct GdnCStatePush {
    float* state;
    float* qn;
    float* kn;
    float* U;
    float* W;
    float* lgo;
    float* delta;
    float* o;
    GdnPArgs a;
};

groupshared float g_cs_red[8 * 16 * 32];
groupshared float g_cs_dl[32 * 32];

// grid (h_v * 4), 256 threads = 8 row blocks of 16 x 32 columns
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gdn_cstate(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform GdnCStatePush pc) {
    const int S = 128, RB = 16;
    const GdnPArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int h = int(tg.x) / 4, hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC;
    const int j = (int(tg.x) % 4) * 32 + lane, i0 = sg * RB;
    float st[16];
    [ForceUnroll] for (int r = 0; r < RB; ++r) st[r] = pc.state[((i0 + r) * hv + h) * S + j];
    for (int c = 0; c < NC; ++c) {
        const int t0 = c * GC, nb = min(GC, T - t0);
        float* Uc = pc.U + (uint64_t(h) * uint(NC) + uint(c)) * uint(GC * S);
        float* Wc = pc.W + (uint64_t(h) * uint(NC) + uint(c)) * uint(GC * S);
        float* lg = pc.lgo + (h * NC + c) * GC;
        for (int pass = 0; pass < 2; ++pass) {
            for (int hf = 0; hf < 2; ++hf) {
                float p[16];
                [ForceUnroll]
                for (int u = 0; u < 16; ++u) {
                    const int t = hf * 16 + u;
                    float s = 0.0f;
                    if (t < nb) {
                        float* v = pass == 0 ? Wc + t * S + i0 : pc.qn + ((t0 + t) * hkn + hk) * S + i0;
                        [ForceUnroll] for (int r = 0; r < RB; ++r) s = fma(v[r], st[r], s);
                    }
                    p[u] = s;
                }
                GroupMemoryBarrierWithGroupSync();
                [ForceUnroll] for (int u = 0; u < 16; ++u) g_cs_red[(sg * 16 + u) * 32 + lane] = p[u];
                GroupMemoryBarrierWithGroupSync();
                for (int u = sg * 2; u < sg * 2 + 2; ++u) {
                    const int t = hf * 16 + u;
                    float s = 0.0f;
                    for (int b = 0; b < 8; ++b) s += g_cs_red[(b * 16 + u) * 32 + lane];
                    if (t < nb) {
                        if (pass == 0) {
                            const float d = Uc[t * S + j] - s;
                            g_cs_dl[t * 32 + lane] = d;
                            pc.delta[((uint64_t(h) * uint(NC) + uint(c)) * uint(GC) + uint(t)) * uint(S) + uint(j)] = d;
                        } else {
                            pc.o[((t0 + t) * hv + h) * S + j] = exp(lg[t]) * s;
                        }
                    } else if (pass == 0) {
                        g_cs_dl[t * 32 + lane] = 0.0f;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        const float lgl = lg[nb - 1], gl = exp(lgl);
        [ForceUnroll] for (int r = 0; r < RB; ++r) st[r] *= gl;
        for (int s = 0; s < nb; ++s) {
            const float f = exp(lgl - lg[s]) * g_cs_dl[s * 32 + lane];
            float* ks = pc.kn + ((t0 + s) * hkn + hk) * S + i0;
            [ForceUnroll] for (int r = 0; r < RB; ++r) st[r] = fma(ks[r], f, st[r]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [ForceUnroll] for (int r = 0; r < RB; ++r) pc.state[((i0 + r) * hv + h) * S + j] = st[r];
}

struct GdnCOutPush {
    float* qn;
    float* kn;
    float* delta;
    float* lgo;
    float* o;
    GdnPArgs a;
};

groupshared float g_co_M[32 * 32];

// grid (h_v, chunks), 128 threads (column j)
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void gdn_cout(uint3 tg : SV_GroupID, uint tidu : SV_GroupIndex, uniform GdnCOutPush pc) {
    const int S = 128;
    const GdnPArgs a = pc.a;
    const int tid = int(tidu);
    const int h = int(tg.x), c = int(tg.y), hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC, t0 = c * GC, nb = min(GC, T - t0);
    float* lg = pc.lgo + (h * NC + c) * GC;
    for (int idx = tid; idx < GC * GC; idx += 128) {
        const int t = idx / GC, s = idx % GC;
        float v = 0.0f;
        if (s <= t && t < nb) {
            float* qt = pc.qn + ((t0 + t) * hkn + hk) * S;
            float* ks = pc.kn + ((t0 + s) * hkn + hk) * S;
            float d = 0.0f;
            for (int i = 0; i < S / 4; ++i) d += dot(gp_ld4(qt + 4 * i), gp_ld4(ks + 4 * i));
            v = d * exp(lg[t] - lg[s]);
        }
        g_co_M[t * GC + s] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    const int j = tid;
    float* dc = pc.delta + (uint64_t(h) * uint(NC) + uint(c)) * uint(GC * S) + uint(j);
    float dv[32];
    [ForceUnroll] for (int s = 0; s < GC; ++s) dv[s] = s < nb ? dc[s * S] : 0.0f;
    const float scale = 1.0f / sqrt(float(S));
    for (int t = 0; t < nb; ++t) {
        float s = pc.o[((t0 + t) * hv + h) * S + j];
        [ForceUnroll] for (int u = 0; u < GC; ++u)
            if (u <= t) s = fma(g_co_M[t * GC + u], dv[u], s);
        pc.o[((t0 + t) * hv + h) * S + j] = s * scale;
    }
}
