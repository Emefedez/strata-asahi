// Feature probe: the constructs the MSL kernels translate to.
struct Args {
    float* x;          // buffer(0)
    float* y;          // buffer(1)
    uint n;            // a GemvArgs-like struct, natural C layout
    uint stride;
};
[[vk::push_constant]] Args a;

[vk::constant_id(0)] const int NTOK = 1;        // a Metal function constant
[vk::constant_id(100)] const int WG = 64;        // threadgroup size from the dispatch

groupshared float red[32];

float sum_rows<let NR : int>(float v) {          // a value generic (MSL template<int NR>)
    float s = 0;
    [ForceUnroll] for (int r = 0; r < NR; ++r) s += v * float(r + 1);
    return s;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void probe_kernel(uint3 tid : SV_DispatchThreadID, uint lid : SV_GroupIndex) {
    uint lane = WaveGetLaneIndex();
    float v = tid.x < a.n ? a.x[tid.x] : 0.0f;
    float s = WaveActiveSum(sum_rows<2>(v)) * float(NTOK);
    float p = WavePrefixSum(v);
    if (lane == 0) red[lid / 32] = s;
    GroupMemoryBarrierWithGroupSync();
    if (lid == 0) {
        float t = 0;
        for (int i = 0; i < WG / 32; ++i) t += red[i];
        a.y[tid.x / WG] = t + p * 0.0f;
    }
}
