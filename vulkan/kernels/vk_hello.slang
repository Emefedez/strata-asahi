// vulkan/kernels/vk_hello.slang - the runtime's own check (tests/vk_hello.cpp): arguments in push constants
// (buffers as addresses, an inline struct after them), a function constant and the threadgroup size.
struct HelloArgs { uint n; float scale; };
struct HelloPush {
    float* x;
    float* y;
    HelloArgs a;
};
[[vk::push_constant]] HelloPush pc;
[vk::constant_id(0)] const int ADD = 0;

[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void vk_hello(uint3 gid : SV_DispatchThreadID) {
    if (gid.x >= pc.a.n) return;
    pc.y[gid.x] = pc.x[gid.x] * pc.a.scale + float(ADD) + WaveActiveSum(1.0f) * 0.0f;
}
