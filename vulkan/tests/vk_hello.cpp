// vulkan/tests/vk_hello.cpp - the Vulkan runtime (mtl_rt_vk.cpp) end to end: compile a Slang kernel at run time,
// pass buffers and an inline struct, set a function constant and the threadgroup size, run serial and graph modes,
// and the in-command-buffer host handshake (run_graph_seg) with host writes between segments.
#include "mtl_rt.h"

#include <cmath>
#include <cstdio>
#include <vector>

struct HelloArgs { uint32_t n; float scale; };

int main() {
    mtl::Context cx;
    std::printf("device: %s, working set %.1f GiB\n", cx.device_name().c_str(), cx.recommended_working_set() / 1073741824.0);
    const int lib = cx.compile({"vk_hello.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline p = cx.pipeline(lib, "vk_hello", mtl::Constants().i(0, 3));
    const uint32_t n = 1000;
    mtl::Buffer x = cx.alloc(n * 4), y = cx.alloc(n * 4);
    for (uint32_t i = 0; i < n; i++) x.as<float>()[i] = float(i);
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(HelloArgs{n, 2.0f})};
    d.grid[0] = (n + 63) / 64;
    d.tg[0] = 64;
    cx.run({d});
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += y.as<float>()[i] != float(i) * 2.0f + 3.0f;
    std::printf("run: %s (%d wrong)\n", bad ? "FAIL" : "ok", bad);

    // run_graph_seg: 8 segments; before segment s+1 the host writes x[0] = s+1, the GPU's next dispatch reads it.
    std::vector<mtl::Dispatch> ds;
    std::vector<int> cuts;
    mtl::Dispatch one = d;
    one.args = {mtl::Arg::B(x), mtl::Arg::B(y, 0), mtl::Arg::V(HelloArgs{1, 1.0f})};
    one.grid[0] = 1;
    std::vector<mtl::Buffer> outs;
    for (int s = 0; s < 8; s++) outs.push_back(cx.alloc(16));
    for (int s = 0; s < 8; s++) {
        mtl::Dispatch e = one;
        e.args[1] = mtl::Arg::B(outs[s]);
        ds.push_back(e);
        if (s < 7) cuts.push_back((int)ds.size());
    }
    struct U { mtl::Buffer* x; } u{&x};
    const double t = cx.run_graph_seg(ds, cuts, [](void* up, int s) { static_cast<U*>(up)->x->as<float>()[0] = float(s + 1); }, &u);
    int bad2 = 0;
    for (int s = 0; s < 8; s++) bad2 += outs[s].as<float>()[0] != float(s) + 3.0f;
    std::printf("run_graph_seg: %s (%d wrong), %.1f us per segment\n", bad2 ? "FAIL" : "ok", bad2, t * 1e6 / 8);
    return bad || bad2;
}
