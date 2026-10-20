// vulkan/kernels/router.slang - metal/kernels/router.metal in Slang (after quant/units/moe/gemv): the router GEMV and
// moe_route in ONE dispatch.  The logits rows as gemv's (LPR = 32 lanes a row, KS = 1), 16 rows a threadgroup of 512
// threads; every threadgroup bumps a device counter after its rows (device-scope fences around it), and the LAST one
// runs moe_route's body with its 512 threads (= n_expert) and resets the counter.  No threadgroup waits for another.
// Buffers: W, xp, s16, logits, args (GemvArgs: n_out = n_expert), ids, w, plan, MoeArgs, counter (one uint, zero).

struct RouterPush {
    uint8_t* W;
    float* xp;
    float* s16;
    float* logits;
    GemvArgs ga;
    int* ids;
    float* w;
    int* plan;
    MoeArgs a;
    Atomic<uint>* counter;
};

groupshared uint g_last;

void router_route<let F : int>(RouterPush pc, uint tgx, uint ngx, uint tid) {
    const int UNIT = f_UNIT(F);
    load_tab<F>(tid, TG_N);
    GroupMemoryBarrierWithGroupSync();
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    const GemvArgs ga = pc.ga;
    // the GEMV: simdgroup sg -> row tg.x * 16 + sg, 32 lanes over its units
    const int row = int(tgx) * 16 + sg;
    const int rr = min(row, int(ga.n_out) - 1);
    uint8_t* rp = pc.W + uint64_t(rr) * ga.row_bytes;
    const int U = int(ga.n_in) / UNIT, s16n = int(ga.n_in) / 16;
    float acc[8];
    [ForceUnroll] for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int u = lane; u < U; u += 32) {
        float q[MAXU];
        float4 sc;
        Dec_run<F>(rp, u, q, sc);
        for (int t = 0; t < NTOK; ++t) {
            float xs[MAXU], xsum[2];
            X_load<F>(pc.xp + t * int(ga.x_stride), pc.s16 + t * s16n, u, xs, xsum);
            acc[t] += unit_combine(UNIT, q, sc, xs, xsum);
        }
    }
    for (int t = 0; t < NTOK; ++t) {
        const float v = lane_sum<32>(acc[t]);
        if (lane == 0 && row < int(ga.n_out)) pc.logits[t * int(ga.y_stride) + row] = v;
    }
    // the last threadgroup to finish routes
    AllMemoryBarrierWithGroupSync();
    if (tid == 0) g_last = pc.counter->add(1u) == ngx - 1 ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    if (g_last == 0) return;
    AllMemoryBarrier();
    route_tok(pc.logits, pc.ids, pc.w, pc.a, sg, lane);
    AllMemoryBarrierWithGroupSync();
    plan_body(pc.ids, pc.w, pc.plan, pc.a, int(tid), TG_N, sg, lane);
    if (tid == 0) pc.counter->store(0u);
}

#define RR(F)                                                                                                         \
    [shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]                                                               \
    void router_route_##F(uint3 tg : SV_GroupID, uint tid : SV_GroupIndex, uniform RouterPush pc) {                  \
        router_route<F>(pc, tg.x, WorkgroupCount().x, tid);                                                          \
    }
RR(30) RR(0)
