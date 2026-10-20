// vulkan/kernels/gdn.slang - metal/kernels/gdn.metal in Slang: one GDN (gated delta-net) layer's recurrence for a
// window of tokens, fused end to end:
//   conv (4 taps, history [channels][3] oldest first) -> SiLU, on q (16 heads), k (16), v (48 heads) of 128
//   q, k: L2 norm as native_gdn_l2_norm (rsqrt(sum/128 + eps/128), then x 1/sqrt(128), two roundings)
//   gate = exp(softplus(alpha + dt) * a), beta = sigmoid(beta)
//   kv_j = sum_i S[i][j] k[i];  delta_j = (v_j - gate * kv_j) * beta;  S = gate * S + k[i] delta_j
//   o_j = (sum_i S[i][j] q[i]) / sqrt(128);  y_j = o_j * rsqrt(mean(o^2) + eps) * gamma_j * sigmoid(z_j)
// State layout as Strata's: S[(i * h_v + h) * 128 + j].  One threadgroup per value head, IB * 128 threads: simdgroup s
// holds columns j = 32 (s % 4) + lane and rows i = RPT (s / 4) .. + RPT of the head's state in registers.
// Commit, conv history ping-pong and the prepared planes as the MSL (see there).

[vk::constant_id(2)] const int GDN_NTOK = 1;

struct GdnArgs {
    uint h_k;
    uint h_v;
    uint channels;
    uint commit;
    float eps;
    uint plane;
};

struct GdnPush {
    float* state;
    float* hist;
    float* qkv;
    float* alpha;
    float* beta;
    float* z;
    float* conv_w;
    float* dt;
    float* ssm_a;
    float* gamma;
    float* y;
    GdnArgs a;
    float* hist_out;
};

groupshared float g_qs[128];
groupshared float g_ks[128];
groupshared float g_vs[128];
groupshared float g_red4[4 * 128];
groupshared float g_red8[8 * 128];
groupshared float g_gpart[32];
groupshared float g_gb[2];
groupshared float g_ys[128];
float red_get<let IB : int>(int i) { return IB == 4 ? g_red4[i] : g_red8[i]; }
void red_set<let IB : int>(int i, float v) {
    if (IB == 4) g_red4[i] = v;
    else g_red8[i] = v;
}

void gdn_step_t<let IB : int>(GdnPush pc, uint3 tg, uint tidu) {
    const int S = 128;
    const int RPT = S / IB;
    const GdnArgs a = pc.a;
    const int tid = int(tidu), sg = tid / 32, lane = tid % 32;
    const int h = int(tg.x), hk = h % int(a.h_k), hv = int(a.h_v);
    const int cb = sg & 3, ib = sg >> 2;
    const int j = 32 * cb + lane, i0 = RPT * ib;
    const float scale = 1.0f / sqrt(float(S));

    int ch = -1;
    if (tid < 128) ch = hk * S + tid;
    else if (tid < 256) ch = int(a.h_k) * S + hk * S + (tid - 128);
    else if (tid < 384) ch = 2 * int(a.h_k) * S + h * S + (tid - 256);
    float hst[3] = {0.0f, 0.0f, 0.0f}, cw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (ch >= 0) {
        [ForceUnroll] for (int k = 0; k < 3; ++k) hst[k] = pc.hist[ch * 3 + k];
        [ForceUnroll] for (int k = 0; k < 4; ++k) cw[k] = pc.conv_w[ch * 4 + k];
    }
    float st[RPT];
    [ForceUnroll] for (int r = 0; r < RPT; ++r) st[r] = pc.state[((i0 + r) * hv + h) * S + j];

    const int commit_at = a.commit == 1u ? GDN_NTOK : ((a.commit >> 8) != 0 ? int(a.commit & 0xFFu) : 0);
    for (int t = 0; t < GDN_NTOK; ++t) {
        float cv = 0.0f;
        if (ch >= 0) {
            const float inp = pc.qkv[t * int(a.channels) + ch];
            const float vals[4] = {hst[0], hst[1], hst[2], inp};
            float s = 0.0f;
            [ForceUnroll] for (int k = 0; k < 4; ++k) s = fma(vals[k], cw[k], s);
            s = s + 0.0f;
            cv = s / (1.0f + exp(-s));
            hst[0] = hst[1];
            hst[1] = hst[2];
            hst[2] = inp;
        }
        const float ssq = WaveActiveSum(cv * cv);
        if (lane == 0) g_gpart[sg] = ssq;
        if (tid == 0) {
            const float v = pc.alpha[t * hv + h] + pc.dt[h];
            const float e = exp(v), u = 1.0f + e;
            const float sp = v > 20.0f ? v : (u == 1.0f ? e : log(u) * e / (u - 1.0f));
            g_gb[0] = exp(sp * pc.ssm_a[h]);
            g_gb[1] = 1.0f / (1.0f + exp(-pc.beta[t * hv + h]));
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < 256) {
            const int base = tid < 128 ? 0 : 4;
            const float tot = (g_gpart[base] + g_gpart[base + 1]) + (g_gpart[base + 2] + g_gpart[base + 3]);
            const float rs = rsqrt(tot / float(S) + a.eps / float(S));
            const float nv = (rs * cv) * scale;
            if (tid < 128) g_qs[tid] = nv;
            else g_ks[tid - 128] = nv;
        } else if (tid < 384) {
            g_vs[tid - 256] = cv;
        }
        GroupMemoryBarrierWithGroupSync();
        const float g = g_gb[0], be = g_gb[1];
        float kv = 0.0f;
        [ForceUnroll] for (int r = 0; r < RPT; ++r) kv = fma(st[r], g_ks[i0 + r], kv);
        red_set<IB>(ib * S + j, kv);
        GroupMemoryBarrierWithGroupSync();
        kv = 0.0f;
        [ForceUnroll] for (int k = 0; k < IB; ++k) kv += red_get<IB>(k * S + j);
        const float delta = (g_vs[j] - g * kv) * be;
        float o = 0.0f;
        [ForceUnroll]
        for (int r = 0; r < RPT; ++r) {
            st[r] = fma(g, st[r], g_ks[i0 + r] * delta);
            o = fma(st[r], g_qs[i0 + r], o);
        }
        GroupMemoryBarrierWithGroupSync();
        red_set<IB>(ib * S + j, o);
        GroupMemoryBarrierWithGroupSync();
        o = 0.0f;
        [ForceUnroll] for (int k = 0; k < IB; ++k) o += red_get<IB>(k * S + j);
        o = o * scale;
        // the whole subgroup takes the same branch (ib is per simdgroup), so the sum is over its 32 lanes
        const float osq = ib == 0 ? WaveActiveSum(o * o) : 0.0f;
        if (ib == 0 && lane == 0) g_gpart[cb] = osq;
        GroupMemoryBarrierWithGroupSync();
        if (ib == 0) {
            const float tot = (g_gpart[0] + g_gpart[1]) + (g_gpart[2] + g_gpart[3]);
            const float rs = rsqrt(tot / float(S) + a.eps);
            const float zz = pc.z[t * hv * S + h * S + j];
            const float yv = ((rs * o) * pc.gamma[j]) * (1.0f / (1.0f + exp(-zz)));
            pc.y[t * hv * S + h * S + j] = yv;
            if (a.plane != 0) {
                float* raw = pc.y + t * hv * S + h * S + j;
                const int jj = j & 15;
                raw[a.plane] = yv * k_gdn_q2s[jj & 7];
                raw[2 * a.plane] = yv * k_gdn_p4s[jj & 3];
                raw[3 * a.plane] = (jj & 1) != 0 ? yv * 0.00390625f : yv;
                g_ys[j] = yv;
            }
        }
        if (t + 1 == commit_at) {
            [ForceUnroll] for (int r = 0; r < RPT; ++r) pc.state[((i0 + r) * hv + h) * S + j] = st[r];
            if (ch >= 0 && (tid >= 256 || h < int(a.h_k)))
                [ForceUnroll] for (int k = 0; k < 3; ++k) pc.hist_out[ch * 3 + k] = hst[k];
        }
        GroupMemoryBarrierWithGroupSync();
        if (a.plane != 0 && ib == 0 && (j & 15) == 0) {
            float s0 = 0.0f, s1 = 0.0f;
            for (int k = 0; k < 16; k += 2) { s0 += g_ys[j + k]; s1 += g_ys[j + k + 1]; }
            pc.y[4 * a.plane + t * (hv * S / 16) + (h * S + j) / 16] = s0 + s1;
        }
    }
}
static const float k_gdn_q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
static const float k_gdn_p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};

[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void gdn_step(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GdnPush pc) { gdn_step_t<4>(pc, tg, tid); }
[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void gdn_step8(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GdnPush pc) { gdn_step_t<8>(pc, tg, tid); }
