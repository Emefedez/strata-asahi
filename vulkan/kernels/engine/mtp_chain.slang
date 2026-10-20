// vulkan/kernels/engine/mtp_chain.slang - metal/src/engine/msl/mtp_chain.metal in Slang (after quant and units): the
// two steps that keep a whole MTP draft round on the GPU:
//   mtp_pick     draft j: the pick's token id (through the draft vocabulary) and its probability
//   mtp_embed_F  the next step's input: the picked token's embedding row dequantized from token_embd (format F)
// mtp_embed reads a row with the unit decoders: weight j of a unit is w = d * (q[j] * 2^-k) - m (see the MSL).

struct PickArgs {
    uint nv;
    uint j;
    uint use_map;
};
struct PickPush {
    float* lg;
    int* pick;
    int* vmap;
    int* dout;
    float* dprob;
    PickArgs a;
};
groupshared float c_part[32];

// one threadgroup of 1024
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void mtp_pick(uint tid : SV_GroupIndex, uniform PickPush pc) {
    const PickArgs a = pc.a;
    const int di = pc.pick[0];
    const float mx = pc.lg[di];
    float s = 0.0f;
    for (uint i = tid; i < a.nv; i += 1024) s += exp(pc.lg[i] - mx);
    s = WaveActiveSum(s);
    if (WaveGetLaneIndex() == 0) c_part[tid / 32] = s;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        float t = 0.0f;
        for (int k = 0; k < 32; ++k) t += c_part[k];
        pc.dout[a.j] = a.use_map != 0 ? pc.vmap[di] : di;
        pc.dprob[a.j] = 1.0f / t;
    }
}

struct EmbArgs {
    uint n;
    uint row_bytes;
    uint j;
};
struct EmbPush {
    uint8_t* W;
    int* dout;
    float* posv;
    float* y;
    EmbArgs a;
};

float kind_scale(int F, int pos) {
    const int k = x_kind(F);
    if (k == XK_Q2) return k_chain_q2s[pos & 7];
    if (k == XK_P4) return k_chain_p4s[pos & 3];
    if (k == XK_P2) return (pos & 1) != 0 ? 0.00390625f : 1.0f;
    return 1.0f;
}
static const float k_chain_q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
static const float k_chain_p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};

// one threadgroup of 256; posv = the floats 0 .. n-1
void mtp_embed<let F : int>(EmbPush pc, uint tid) {
    const int UNIT = f_UNIT(F), H = UNIT / 2;
    load_tab<F>(tid, 256);
    GroupMemoryBarrierWithGroupSync();
    const EmbArgs a = pc.a;
    uint8_t* row = pc.W + uint64_t(pc.dout[a.j]) * a.row_bytes;
    const int U = int(a.n) / UNIT;
    for (int u = int(tid); u < U; u += 256) {
        float q[MAXU], ps[MAXU], xsum[2];
        float4 sc;
        Dec_run<F>(row, u, q, sc);
        X_load<F>(pc.posv, pc.posv, u, ps, xsum);
        [unroll]
        for (int j = 0; j < f_UNIT(F); ++j) {
            const int pos = int(ps[j]);
            const float d = j < H ? sc.x : sc.y;
            const float m = (UNIT == 16 || j < 16) ? sc.z : sc.w;
            pc.y[pos] = d * (q[j] * kind_scale(F, pos)) - m;
        }
    }
}

#define EMB(F)                                                                                                        \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void mtp_embed_##F(uint tid : SV_GroupIndex, uniform EmbPush pc) { mtp_embed<F>(pc, tid); }
// T_BF16 30, T_Q8_0 8, T_Q4_0 2, T_Q5_0 6, T_IQ4_NL 20, T_IQ4_XS 23, T_Q2_0 42, T_IQ3_S 21, T_Q3_K 11, T_Q4_K 12,
// T_Q5_K 13, T_Q6_K 14, T_IQ3_XXS 18
EMB(30) EMB(8) EMB(2) EMB(6) EMB(20) EMB(23) EMB(42) EMB(21) EMB(11) EMB(12) EMB(13) EMB(14) EMB(18)
