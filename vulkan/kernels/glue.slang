// vulkan/kernels/glue.slang - metal/kernels/glue.metal in Slang: small elementwise steps between the big kernels.
//   swiglu_prep    h = silu(g) * u for the shared expert, written as prepared activations (xprep layout)
//   moe_combine_sh bo[t] = sum_k ye[entry(t, k)] + sigmoid(sg[t]) * ysh[t]   (k in order)
//   slc_touch      reads a weight range so it sits in the SLC

struct GlueArgs {
    uint n;
    uint n_tok;
    uint n_embd;
    uint topk;
};

groupshared float g_sw[16];

struct SwigluPush {
    float* g;
    float* u;
    float* xp;
    GlueArgs a;
};
static const float k_q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
static const float k_p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};

// grid (n / 16, n_tok), 16 threads: one group of 16 values
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void swiglu_prep(uint3 tg : SV_GroupID, uint j : SV_GroupIndex, uniform SwigluPush pc) {
    const GlueArgs a = pc.a;
    const uint t = tg.y, d = tg.x * 16 + j, plane = a.n_tok * a.n;
    const float gv = pc.g[t * a.n + d];
    const float h = gv / (1.0f + exp(-gv)) * pc.u[t * a.n + d];
    float* raw = pc.xp + t * a.n + d;
    raw[0] = h;
    raw[plane] = h * k_q2s[j & 7];
    raw[2 * plane] = h * k_p4s[j & 3];
    raw[3 * plane] = (j & 1) != 0 ? h * 0.00390625f : h;
    g_sw[j] = h;
    GroupMemoryBarrierWithGroupSync();
    if (j == 0) {
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += g_sw[k]; s1 += g_sw[k + 1]; }
        pc.xp[4 * plane + t * (a.n / 16) + tg.x] = s0 + s1;
    }
}

struct CombineShPush {
    float* ye;
    int* plan;
    float* ysh;
    float* sg;
    float* bo;
    GlueArgs a;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void moe_combine_sh(uint gid : SV_DispatchThreadID, uniform CombineShPush pc) {
    const GlueArgs a = pc.a;
    const uint t = gid / a.n_embd, k = gid % a.n_embd;
    if (t >= a.n_tok) return;
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += pc.ye[pc.plan[P_TOKENT + t * a.topk + j] * int(a.n_embd) + k];
    const float gs = 1.0f / (1.0f + exp(-pc.sg[t]));
    pc.bo[gid] = fma(gs, pc.ysh[gid], s);
}

struct SlcPush {
    uint4* src;
    uint* sink;
    uint n16;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void slc_touch(uint gid : SV_DispatchThreadID, uniform SlcPush pc) {
    uint s = 0;
    for (uint k = 0; k < 4; ++k) {
        const uint j = (gid >> 8) * 1024 + k * 256 + (gid & 255);
        if (j < pc.n16) {
            const uint4 v = pc.src[j];
            s ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (s == 0x9e3779b9u) pc.sink[0] = s;
}
