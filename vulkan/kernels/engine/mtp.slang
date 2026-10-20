// vulkan/kernels/engine/mtp.slang - metal/src/engine/msl/mtp.metal in Slang: the small steps of the MTP draft layer's
// input branches around the shared kernels (GEMVs, the hyper-connection read, QSA, MoE, the head):
//   mtp_rms      y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w        rows of width n
//   mtp_xprep    xprep of rows read with a stride: the prepared activation of one stream of rows
//   mtp_combine  R[t][c][d] = h[c][t][d] + e[t][d]                  the streams of fc_hidden plus fc_embedding

struct RmsArgs { uint n; float eps; };
struct RmsPush {
    float* x;
    float* w;
    float* y;
    RmsArgs a;
};
groupshared float m_part[8];

// grid (rows), 256 threads
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mtp_rms(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform RmsPush pc) {
    float* xr = pc.x + tg.x * pc.a.n;
    float ss = 0.0f;
    for (uint d = tid; d < pc.a.n; d += 256) ss = fma(xr[d], xr[d], ss);
    ss = WaveActiveSum(ss);
    if (WaveGetLaneIndex() == 0) m_part[tid / 32] = ss;
    GroupMemoryBarrierWithGroupSync();
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += m_part[i];
    const float rs = rsqrt(tot / float(pc.a.n) + pc.a.eps);
    float* yr = pc.y + tg.x * pc.a.n;
    for (uint d = tid; d < pc.a.n; d += 256) yr[d] = xr[d] * rs * pc.w[d];
}

struct XpArgs { uint n; uint n_tok; uint plane_tok; uint stride; };
struct XpPush {
    float* x;
    float* xp;
    XpArgs a;
};
static const float k_mtp_q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
static const float k_mtp_p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};

// one thread per 16 values of one row: row t at x + t * stride; out planes of plane_tok rows
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mtp_xprep(uint gid : SV_DispatchThreadID, uniform XpPush pc) {
    const XpArgs a = pc.a;
    const uint per = a.n / 16;
    if (gid >= per * a.n_tok) return;
    const uint t = gid / per, i = gid % per;
    float* src = pc.x + t * a.stride + 16 * i;
    const uint plane = a.plane_tok * a.n;
    float* raw = pc.xp + t * a.n + 16 * i;
    float v[16];
    [ForceUnroll] for (int j = 0; j < 16; ++j) v[j] = src[j];
    float s0 = 0, s1 = 0;
    [ForceUnroll] for (int j = 0; j < 16; j += 2) { s0 += v[j]; s1 += v[j + 1]; }
    [ForceUnroll]
    for (int j = 0; j < 16; ++j) {
        raw[j] = v[j];
        raw[plane + j] = v[j] * k_mtp_q2s[j & 7];
        raw[2 * plane + j] = v[j] * k_mtp_p4s[j & 3];
        raw[3 * plane + j] = (j & 1) != 0 ? v[j] * 0.00390625f : v[j];
    }
    pc.xp[4 * plane + t * per + i] = s0 + s1;
}

struct CombArgs { uint n; uint hc; uint n_tok; uint h_stride; };
struct CombPush {
    float* h;
    float* e;
    float* R;
    CombArgs a;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mtp_combine(uint gid : SV_DispatchThreadID, uniform CombPush pc) {
    const CombArgs a = pc.a;
    const uint per = a.hc * a.n;
    if (gid >= a.n_tok * per) return;
    const uint t = gid / per, c = (gid % per) / a.n, d = gid % a.n;
    pc.R[gid] = pc.h[c * a.h_stride + t * a.n + d] + pc.e[t * a.n + d];
}
