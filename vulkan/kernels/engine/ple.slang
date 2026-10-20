// vulkan/kernels/engine/ple.slang - metal/src/engine/msl/ple.metal in Slang: the PLE block after its two projections,
// for a window of T rows (the CUDA engine's ple.cu math; its block sums are double there, float here):
//   key = grouped_norm(pkey, norm_key), query = grouped_norm(R, norm_query)
//   gate[c] = sigmoid(sign(s) sqrt(max(|s|, 1e-6))),  s = key[c] . query[c] / sqrt(n_embd)
//   gated = value[d] * gate[c];   normalized = grouped_norm(gated, norm_conv)
//   conv[c] = silu(sum_k kW[k + 4c] * row(t - (3 - k) * 3)[c]);   R += gated + conv

struct PleArgs {
    uint n_embd;
    uint hc;
    float eps;
};

groupshared float p_part[8];

float tg_sum(float v, uint tid) {
    v = WaveActiveSum(v);
    if (WaveGetLaneIndex() == 0) p_part[tid / 32] = v;
    GroupMemoryBarrierWithGroupSync();
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) t += p_part[i];
    GroupMemoryBarrierWithGroupSync();
    return t;
}

struct GnormPush {
    float* x;
    float* w;
    float* y;
    PleArgs a;
};
// grid (hc, T), 256 threads: y[t][c] = grouped_norm(x[t][c]) with w[c]
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void ple_gnorm(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GnormPush pc) {
    const PleArgs a = pc.a;
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    float* xc = pc.x + (t * a.hc + c) * n;
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) ss += xc[d] * xc[d];
    const float mean = tg_sum(ss, tid) / float(n);
    const float scale = 1.0f / sqrt(mean + a.eps);
    float* yc = pc.y + (t * a.hc + c) * n;
    for (uint d = tid; d < n; d += 256) yc[d] = xc[d] * scale * pc.w[c * n + d];
}

struct GatePush {
    float* key;
    float* query;
    float* value;
    float* w_conv;
    float* gated;
    float* norm_w;
    PleArgs a;
};
// grid (hc, T), 256 threads: the gate, then gated = value * gate and normalized = grouped_norm(gated, norm_conv)
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void ple_gate_norm(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform GatePush pc) {
    const PleArgs a = pc.a;
    const uint c = tg.x, t = tg.y, n = a.n_embd, row = (t * a.hc + c) * n;
    float acc = 0.0f;
    for (uint d = tid; d < n; d += 256) acc += pc.key[row + d] * pc.query[row + d];
    const float s = tg_sum(acc, tid) * (1.0f / sqrt(float(n)));
    const float mag = sqrt(max(abs(s), 1e-6f));
    const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
    const float gate = 1.0f / (1.0f + exp(-(sgn * mag)));
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        const float g = pc.value[t * n + d] * gate;
        pc.gated[row + d] = g;
        ss += g * g;
    }
    const float mean = tg_sum(ss, tid) / float(n);
    const float scale = 1.0f / sqrt(mean + a.eps);
    for (uint d = tid; d < n; d += 256) pc.norm_w[row + d] = pc.gated[row + d] * scale * pc.w_conv[c * n + d];
}

struct ConvPush {
    float* hist;
    float* norm_w;
    half* kw;
    float* gated;
    float* R;
    PleArgs a;
};
// grid (hc * n_embd / 256, T), 256 threads: the conv over [history | the window's rows], SiLU, and the residual
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void ple_conv_add(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform ConvPush pc) {
    const PleArgs a = pc.a;
    const uint ch = tg.x * 256 + tid, t = tg.y, hcd = a.hc * a.n_embd;
    if (ch >= hcd) return;
    float acc = 0.0f;
    [ForceUnroll]
    for (int k = 0; k < 4; ++k) {
        const int p = int(t) - (3 - k) * 3;
        const float v = p >= 0 ? pc.norm_w[uint(p) * hcd + ch] : pc.hist[uint(9 + p) + 9 * ch];
        acc += float(pc.kw[k + 4 * ch]) * v;
    }
    const float conv = acc / (1.0f + exp(-acc));
    float* r = pc.R + t * hcd;
    r[ch] = r[ch] + pc.gated[t * hcd + ch] + conv;
}
