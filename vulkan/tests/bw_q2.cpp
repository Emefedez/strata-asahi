// vulkan/tests/bw_q2.cpp - the Q2_0 weight reads of the GEMV, one pattern at a time: GB/s of weight bytes.
//
// The weights are a Q2_0 block array (18 bytes a block, 4 units a block); every lane walks the units with a
// grid-stride loop, the way the GEMV's lanes walk a row's units.
#include "mtl_rt.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {
double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? (size_t) atoi(argv[1]) : 512;
    const int reps = argc > 2 ? atoi(argv[2]) : 8;
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    const int lib = cx.compile({"bw_q2.slang"}, "#define NO_GGML_COMMON 1\n");

    const size_t bytes = (size_t) mib << 20;
    mtl::Buffer src = cx.alloc(bytes);
    std::memset(src.ptr, 0x5a, bytes);
    mtl::Buffer dst = cx.alloc(1 << 20);
    const uint32_t n_units = (uint32_t) (bytes / 18 * 4);

    struct BwArgs { uint32_t n_units, n_threads, mode; };
    const char* names[] = {"4B codes at +2 + 2B scale (today)", "4B codes at +2 + 4B scale",
                           "2 x 4B codes + 4B scale",          "16B codes + 4B scale",
                           "2 x 4B codes, no scale",            "2 x 16B per block"};
    for (uint32_t mode = 0; mode < 6; mode++) {
        double best = 1e30;
        uint32_t wg = 256, groups = 1024;
        for (uint32_t g : {256u, 1024u, 4096u}) {
            mtl::Pipeline p = cx.pipeline(lib, "bw_q2");
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(src), mtl::Arg::B(dst), mtl::Arg::V(BwArgs{n_units, wg * g, mode})};
            d.grid[0] = g;
            d.tg[0] = wg;
            std::vector<mtl::Dispatch> ds;
            for (int r = 0; r < reps; r++) ds.push_back(d);
            cx.warm(ds, 0.05);
            std::vector<double> t;
            for (int i = 0; i < 5; i++) t.push_back(cx.run(ds) / reps);
            std::sort(t.begin(), t.end());
            const double m = t[t.size() / 2];
            if (m < best) best = m;
        }
        // the useful bytes: 18 per 4 units
        const double useful = (double) n_units / 4.0 * 18.0;
        std::printf("  %-32s %8.1f GB/s\n", names[mode], useful / best / 1e9);
    }
    return 0;
}
