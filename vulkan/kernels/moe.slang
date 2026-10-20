// vulkan/kernels/moe.slang - metal/kernels/moe.metal in Slang: the routed experts of one layer for a window of 1..8
// tokens, grouped by expert.  Same entry points, buffers (in argument order, as one push-constant struct) and function
// constants as the MSL:
//   moe_plan    one threadgroup: the window's top-k ids -> groups (distinct experts, by id) with their entries
//               (token, weight), each (token, k)'s entry index, and the indirect grid sizes of the next two kernels
//   moe_route   the router's tail (softmax / top-10 / weights) for the window, then the plan
//   moe_gu      per group, a block of FFN rows: h = silu(gate . x) * (up . x) per entry
//   moe_down    per group, a block of output rows: ye[entry] = weight * (down . h[entry])
//   moe_combine y[token] = sum over k of ye[entry(token, k)], in k order

#define P_NG 0
#define P_IGU 4
#define P_IDN 8
#define P_MAXE 128
#define P_GEXP 16
#define P_GSTART (P_GEXP + P_MAXE)
#define P_ETOK (P_GSTART + P_MAXE + 4)
#define P_EW (P_ETOK + P_MAXE)
#define P_TOKENT (P_EW + P_MAXE)
#define P_INTS (P_TOKENT + P_MAXE)

struct MoeArgs {
    uint64_t gate_off;
    uint64_t up_off;
    uint64_t down_off;
    uint64_t gu_stride;
    uint64_t d_stride;
    uint gu_row;
    uint d_row;
    uint n_embd;
    uint n_ff;
    uint n_tok;
    uint topk;
    uint n_expert;
    uint gx_gu;
    uint gx_dn;
    uint pad;
};

// The SSD-streaming arena (function constant 20): expert e lies in an arena slot, smap[e] = (chunk << 28) | its 16 KiB
// page in that chunk; the chunks are W and W1..W3.  The same block is in moe_mm.slang.
#ifndef STRATA_ARENA_SLANG
#define STRATA_ARENA_SLANG
[vk::constant_id(20)] const bool kArena = false;
uint8_t* arena_slot(uint v, uint8_t* W, uint8_t* W1, uint8_t* W2, uint8_t* W3) {
    const uint c = v >> 28;
    uint8_t* b = c == 0 ? W : (c == 1 ? W1 : (c == 2 ? W2 : W3));
    return b + uint64_t(v & 0x0FFFFFFFu) * 16384ull;
}
#endif

groupshared int s_grp[32];
groupshared int s_ent[32];

void plan_body(int* ids, float* w, int* plan, MoeArgs a, int tid, int nth, int sg, int lane) {
    const int ne = int(a.n_tok * a.topk);
    const int e = tid;   // one thread per expert id (nth == n_expert)
    int cnt = 0;
    if (e < int(a.n_expert))
        for (int i = 0; i < ne; ++i) cnt += ids[i] == e ? 1 : 0;
    const int has = cnt > 0 ? 1 : 0;
    const int pg = simd_prefix_exclusive_sum(has), pe = simd_prefix_exclusive_sum(cnt);
    if (lane == 31) { s_grp[sg] = pg + has; s_ent[sg] = pe + cnt; }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        int og = 0, oe = 0;
        const int nsg = (nth + 31) / 32;
        for (int i = 0; i < nsg; ++i) {
            const int g = s_grp[i], n = s_ent[i];
            s_grp[i] = og;
            s_ent[i] = oe;
            og += g;
            oe += n;
        }
        plan[P_NG] = og;
        plan[P_IGU] = int(a.gx_gu); plan[P_IGU + 1] = og; plan[P_IGU + 2] = 1;
        plan[P_IDN] = int(a.gx_dn); plan[P_IDN + 1] = og; plan[P_IDN + 2] = 1;
        plan[P_GSTART + og] = oe;
    }
    GroupMemoryBarrierWithGroupSync();
    if (has != 0) {
        const int gi = s_grp[sg] + pg, off = s_ent[sg] + pe;
        plan[P_GEXP + gi] = e;
        plan[P_GSTART + gi] = off;
        int j = 0;
        for (int i = 0; i < ne; ++i)
            if (ids[i] == e) {
                plan[P_ETOK + off + j] = i / int(a.topk);
                plan[P_EW + off + j] = asint(w[i]);
                plan[P_TOKENT + i] = off + j;
                ++j;
            }
    }
}

struct PlanPush {
    int* ids;
    float* w;
    int* plan;
    MoeArgs a;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void moe_plan(uint tid : SV_GroupIndex, uniform PlanPush pc) {
    plan_body(pc.ids, pc.w, pc.plan, pc.a, int(tid), TG_N, int(tid) / 32, int(WaveGetLaneIndex()));
}

// simdgroup sg routes token sg exactly as native_router.cu: softmax over the 512 logits, top 10 by probability (equal
// values: the lower expert index), weights = p / max(sum of the selected, 2^-14), the sum taken by the winning lane in
// selection order and reduced with the same xor tree.
void route_tok(float* logits, int* ids, float* w, MoeArgs a, int sg, int lane) {
    if (sg >= int(a.n_tok)) return;
    float* l = logits + sg * 512;
    float v[16];
    [ForceUnroll] for (int i = 0; i < 16; ++i) v[i] = l[lane + 32 * i];
    float mx = -INFINITY;
    [ForceUnroll] for (int i = 0; i < 16; ++i) mx = max(mx, v[i]);
    for (uint m = 16; m > 0; m >>= 1) mx = max(mx, shfl_xor(mx, m));
    float sum = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < 16; ++i) {
        v[i] = exp(v[i] - mx);
        sum += v[i];
    }
    for (uint m = 16; m > 0; m >>= 1) sum += shfl_xor(sum, m);
    const float rcp = 1.0f / sum;
    [ForceUnroll]
    for (int i = 0; i < 16; ++i) {
        v[i] *= rcp;
        if (isnan(v[i])) v[i] = -FLT_MAX;
    }
    float selected = 0.0f, ssum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
        [ForceUnroll]
        for (int i = 1; i < 16; ++i)
            if (v[i] > best) { best = v[i]; ex = lane + 32 * i; }
        for (uint m = 16; m > 0; m >>= 1) {
            const float ob = shfl_xor(best, m);
            const int oe = shfl_xor(ex, m);
            if (ob > best || (ob == best && oe < ex)) { best = ob; ex = oe; }
        }
        if ((ex & 31) == lane) {
            [ForceUnroll] for (int i = 0; i < 16; ++i) if (i == ex / 32) v[i] = -INFINITY;
            ids[sg * 10 + rank] = ex;
            ssum += best;
        }
        if (rank == lane) selected = best;
    }
    for (uint m = 16; m > 0; m >>= 1) ssum += shfl_xor(ssum, m);
    ssum = max(ssum, 6.103515625e-5f);
    const float inv = 1.0f / ssum;
    if (lane < 10) w[sg * 10 + lane] = selected * inv;
}

struct RoutePush {
    float* logits;
    int* ids;
    float* w;
    int* plan;
    MoeArgs a;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void moe_route(uint tid : SV_GroupIndex, uniform RoutePush pc) {
    const int sg = int(tid) / 32, lane = int(WaveGetLaneIndex());
    route_tok(pc.logits, pc.ids, pc.w, pc.a, sg, lane);
    AllMemoryBarrierWithGroupSync();
    plan_body(pc.ids, pc.w, pc.plan, pc.a, int(tid), TG_N, sg, lane);
}

// ---- gate + up + SwiGLU -------------------------------------------------------------------------------------------
void gu_body<let F : int, let LPR : int, let NR : int, let NE : int>(uint8_t* g0, uint8_t* u0, float* x, float* xs16,
                                                                     float* h, int* etok, int c, int row0, int li, MoeArgs a) {
    const int UNIT = f_UNIT(F);
    float ag[NR][NE], au[NR][NE];
    [ForceUnroll] for (int r = 0; r < NR; ++r)
        [ForceUnroll] for (int t = 0; t < NE; ++t) { ag[r][t] = 0.0f; au[r][t] = 0.0f; }
    const int U = int(a.n_embd) / UNIT;
    for (int u = li; u < U; u += LPR) {
        float qg[NR][MAXU], qu[NR][MAXU];
        float4 cg[NR], cu[NR];
        [ForceUnroll]
        for (int r = 0; r < NR; ++r) {
            Dec_run<F>(g0 + r * a.gu_row, u, qg[r], cg[r]);
            Dec_run<F>(u0 + r * a.gu_row, u, qu[r], cu[r]);
        }
        [ForceUnroll]
        for (int t = 0; t < NE; ++t) {
            const int tok = etok[t];
            float xs[MAXU], xsum[2];
            X_load<F>(x + tok * int(a.n_embd), xs16 + tok * int(a.n_embd / 16), u, xs, xsum);
            [ForceUnroll]
            for (int r = 0; r < NR; ++r) {
                ag[r][t] += unit_combine(UNIT, qg[r], cg[r], xs, xsum);
                au[r][t] += unit_combine(UNIT, qu[r], cu[r], xs, xsum);
            }
        }
    }
    [ForceUnroll]
    for (int r = 0; r < NR; ++r)
        [ForceUnroll]
        for (int t = 0; t < NE; ++t) {
            const float gv = lane_sum<LPR>(ag[r][t]), uv = lane_sum<LPR>(au[r][t]);
            if (li == 0) h[(c + t) * int(a.n_ff) + row0 + r] = gv / (1.0f + exp(-gv)) * uv;
        }
}

// bench probe (function constant 22): 1 = no table load (wrong values for the i-quants)
[vk::constant_id(22)] const int MOE_PROBE = 0;

struct MoePush {
    uint8_t* W;
    float* x;      // moe_down: h
    float* h;      // moe_down: ye
    int* plan;
    MoeArgs a;
    float* xs16;   // moe_down: hs16
    uint* smap;
    uint8_t* W1;
    uint8_t* W2;
    uint8_t* W3;
};

void moe_gu<let F : int, let LPR : int, let NR : int, let SG : int>(MoePush pc, uint3 tg, uint tid) {
    const int RG = 32 / LPR;
    if (MOE_PROBE != 1) {
        load_tab<F>(tid, SG * 32);
        GroupMemoryBarrierWithGroupSync();
    }
    const MoeArgs a = pc.a;
    const int g = int(tg.y);
    if (g >= pc.plan[P_NG]) return;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const uint64_t ex = uint64_t(pc.plan[P_GEXP + g]);
    const int c = pc.plan[P_GSTART + g], ne = pc.plan[P_GSTART + g + 1] - c;
    int* etok = pc.plan + P_ETOK + c;
    uint8_t* E = kArena ? arena_slot(pc.smap[ex], pc.W, pc.W1, pc.W2, pc.W3) : pc.W + ex * a.gu_stride;
    uint8_t* g0 = E + a.gate_off + uint64_t(row0) * a.gu_row;
    uint8_t* u0 = E + a.up_off + uint64_t(row0) * a.gu_row;
    switch (ne) {
        case 1: gu_body<F, LPR, NR, 1>(g0, u0, pc.x, pc.xs16, pc.h, etok, c, row0, li, a); break;
        case 2: gu_body<F, LPR, NR, 2>(g0, u0, pc.x, pc.xs16, pc.h, etok, c, row0, li, a); break;
        case 3: gu_body<F, LPR, NR, 3>(g0, u0, pc.x, pc.xs16, pc.h, etok, c, row0, li, a); break;
        case 4: gu_body<F, LPR, NR, 4>(g0, u0, pc.x, pc.xs16, pc.h, etok, c, row0, li, a); break;
        default:
            for (int k = 0; k < ne; k += 4) {
                const int m = min(4, ne - k);
                if (m == 4) gu_body<F, LPR, NR, 4>(g0, u0, pc.x, pc.xs16, pc.h, etok + k, c + k, row0, li, a);
                else if (m == 3) gu_body<F, LPR, NR, 3>(g0, u0, pc.x, pc.xs16, pc.h, etok + k, c + k, row0, li, a);
                else if (m == 2) gu_body<F, LPR, NR, 2>(g0, u0, pc.x, pc.xs16, pc.h, etok + k, c + k, row0, li, a);
                else gu_body<F, LPR, NR, 1>(g0, u0, pc.x, pc.xs16, pc.h, etok + k, c + k, row0, li, a);
            }
            break;
    }
}

// ---- down ---------------------------------------------------------------------------------------------------------
void dn_body<let F : int, let LPR : int, let NR : int, let NE : int>(uint8_t* d0, float* h, float* hs16, float* ye,
                                                                     int* plan, int c, int row0, int li, MoeArgs a) {
    const int UNIT = f_UNIT(F);
    float ac[NR][NE];
    [ForceUnroll] for (int r = 0; r < NR; ++r)
        [ForceUnroll] for (int t = 0; t < NE; ++t) ac[r][t] = 0.0f;
    const int U = int(a.n_ff) / UNIT;
    for (int u = li; u < U; u += LPR) {
        float q[NR][MAXU];
        float4 sc[NR];
        [ForceUnroll] for (int r = 0; r < NR; ++r) Dec_run<F>(d0 + r * a.d_row, u, q[r], sc[r]);
        [ForceUnroll]
        for (int t = 0; t < NE; ++t) {
            float xs[MAXU], xsum[2];
            X_load<F>(h + (c + t) * int(a.n_ff), hs16 + (c + t) * int(a.n_ff / 16), u, xs, xsum);
            [ForceUnroll] for (int r = 0; r < NR; ++r) ac[r][t] += unit_combine(UNIT, q[r], sc[r], xs, xsum);
        }
    }
    [ForceUnroll]
    for (int r = 0; r < NR; ++r)
        [ForceUnroll]
        for (int t = 0; t < NE; ++t) {
            const float v = lane_sum<LPR>(ac[r][t]);
            if (li == 0) ye[(c + t) * int(a.n_embd) + row0 + r] = asfloat(plan[P_EW + c + t]) * v;
        }
}

void moe_down<let F : int, let LPR : int, let NR : int, let SG : int>(MoePush pc, uint3 tg, uint tid) {
    const int RG = 32 / LPR;
    load_tab<F>(tid, SG * 32);
    GroupMemoryBarrierWithGroupSync();
    const MoeArgs a = pc.a;
    const int g = int(tg.y);
    if (g >= pc.plan[P_NG]) return;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const uint64_t ex = uint64_t(pc.plan[P_GEXP + g]);
    const int c = pc.plan[P_GSTART + g], ne = pc.plan[P_GSTART + g + 1] - c;
    uint8_t* d0 = (kArena ? arena_slot(pc.smap[ex], pc.W, pc.W1, pc.W2, pc.W3) : pc.W + ex * a.d_stride) + a.down_off +
                  uint64_t(row0) * a.d_row;
    float* h = pc.x;
    float* ye = pc.h;
    float* hs16 = pc.xs16;
    switch (ne) {
        case 1: dn_body<F, LPR, NR, 1>(d0, h, hs16, ye, pc.plan, c, row0, li, a); break;
        case 2: dn_body<F, LPR, NR, 2>(d0, h, hs16, ye, pc.plan, c, row0, li, a); break;
        case 3: dn_body<F, LPR, NR, 3>(d0, h, hs16, ye, pc.plan, c, row0, li, a); break;
        case 4: dn_body<F, LPR, NR, 4>(d0, h, hs16, ye, pc.plan, c, row0, li, a); break;
        default:
            for (int k = 0; k < ne; k += 4) {
                const int m = min(4, ne - k);
                if (m == 4) dn_body<F, LPR, NR, 4>(d0, h, hs16, ye, pc.plan, c + k, row0, li, a);
                else if (m == 3) dn_body<F, LPR, NR, 3>(d0, h, hs16, ye, pc.plan, c + k, row0, li, a);
                else if (m == 2) dn_body<F, LPR, NR, 2>(d0, h, hs16, ye, pc.plan, c + k, row0, li, a);
                else dn_body<F, LPR, NR, 1>(d0, h, hs16, ye, pc.plan, c + k, row0, li, a);
            }
            break;
    }
}

struct CombinePush {
    float* ye;
    float* y;
    int* plan;
    MoeArgs a;
};
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void moe_combine(uint gid : SV_DispatchThreadID, uniform CombinePush pc) {
    const MoeArgs a = pc.a;
    const uint t = gid / a.n_embd, k = gid % a.n_embd;
    if (t >= a.n_tok) return;
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += pc.ye[pc.plan[P_TOKENT + t * a.topk + j] * int(a.n_embd) + k];
    pc.y[gid] = s;
}

// ---- instantiations: moe_gu_<type>_<LPR>_<NR>_<SG>, moe_down_<type>_<LPR>_<NR>_<SG> -------------------------------
#define GU(F, L, R, S)                                                                                                \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void moe_gu_##F##_##L##_##R##_##S(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform MoePush pc) {         \
        moe_gu<F, L, R, S>(pc, tg, tid);                                                                             \
    }
#define DN(F, L, R, S)                                                                                                \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void moe_down_##F##_##L##_##R##_##S(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform MoePush pc) {       \
        moe_down<F, L, R, S>(pc, tg, tid);                                                                           \
    }
#define GU_SET(F) GU(F, 32, 1, 4) GU(F, 32, 2, 4) GU(F, 32, 4, 4) GU(F, 32, 2, 2) GU(F, 32, 2, 8) GU(F, 32, 1, 8) \
                  GU(F, 16, 1, 4) GU(F, 16, 2, 4) GU(F, 8, 1, 4) GU(F, 8, 2, 4) GU(F, 8, 1, 2)
#define DN_SET(F) DN(F, 8, 1, 4) DN(F, 8, 2, 4) DN(F, 8, 4, 4) DN(F, 8, 1, 8) DN(F, 8, 2, 8) DN(F, 4, 1, 4) DN(F, 4, 2, 4) \
                  DN(F, 4, 4, 4) DN(F, 16, 1, 4) DN(F, 16, 2, 4) DN(F, 32, 1, 4) DN(F, 2, 2, 4)
// the type codes (MSL host names): T_Q2_0 42, T_IQ3_XXS 18, T_IQ2_XXS 16, T_IQ2_XS 17, T_IQ2_S 22, T_IQ3_S 21,
// T_IQ4_XS 23, T_Q4_K 12, T_Q5_K 13, T_Q6_K 14, T_Q8_0 8, T_Q5_1 7, T_Q2_0R 142, T_IQ4_NL 20
GU_SET(42) GU_SET(18) GU_SET(16) GU_SET(17) GU_SET(22) GU_SET(21) GU_SET(23) GU_SET(12) GU_SET(13) GU_SET(14) GU_SET(8)
DN_SET(7) DN_SET(8) DN_SET(42) DN_SET(142) DN_SET(20)
