// vulkan/kernels/sample.slang - metal/kernels/sample.metal in Slang: greedy sampling over the LM head's logits (the
// largest logit, ties to the lowest token id, NaN never wins).  argmax_part (grid (parts, n_tok), 256 threads) writes
// one (value, id) pair per threadgroup; argmax_final (grid (n_tok), 256 threads) reduces them in order.

struct ArgArgs {
    uint n_vocab;
    uint parts;
};

void better(inout float bv, inout int bi, float v, int i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

groupshared float g_sv[8];
groupshared int g_si[8];

void tg_argmax(inout float bv, inout int bi, uint sg) {
    for (uint m = 16; m > 0; m >>= 1) {
        const float ov = shfl_xor(bv, m);
        const int oi = shfl_xor(bi, m);
        better(bv, bi, ov, oi);
    }
    if (WaveGetLaneIndex() == 0) { g_sv[sg] = bv; g_si[sg] = bi; }
    GroupMemoryBarrierWithGroupSync();
    bv = -INFINITY;
    bi = 0x7fffffff;
    for (int k = 0; k < 8; ++k) better(bv, bi, g_sv[k], g_si[k]);
}

struct PartPush {
    float* logits;
    float* pv;
    int* pi;
    ArgArgs a;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void argmax_part(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform PartPush pc) {
    const ArgArgs a = pc.a;
    const uint t = tg.y, per = (a.n_vocab + a.parts - 1) / a.parts;
    const uint b0 = tg.x * per, b1 = min(b0 + per, a.n_vocab);
    float* l = pc.logits + t * a.n_vocab;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint i = b0 + tid; i < b1; i += 256) {
        const float v = l[i];
        if (!isnan(v)) better(bv, bi, v, int(i));
    }
    tg_argmax(bv, bi, tid / 32);
    if (tid == 0) { pc.pv[t * a.parts + tg.x] = bv; pc.pi[t * a.parts + tg.x] = bi; }
}

struct FinalPush {
    float* pv;
    int* pi;
    int* out;
    ArgArgs a;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void argmax_final(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform FinalPush pc) {
    const ArgArgs a = pc.a;
    const uint t = tg.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint k = tid; k < a.parts; k += 256) better(bv, bi, pc.pv[t * a.parts + k], pc.pi[t * a.parts + k]);
    tg_argmax(bv, bi, tid / 32);
    if (tid == 0) pc.out[t] = bi;
}
