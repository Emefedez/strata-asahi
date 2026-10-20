// vulkan/tests/vk_ptr.cpp - can the Slang kernels turn a buffer pointer into an integer (to know its alignment)?
//
// The weight decoders need an address's alignment when a field has to be read with a 4-byte load.  This test runs
// the kernel on a buffer at eight offsets and checks that it finds the 4-byte aligned word at or below the buffer
// (the buffer holds a known pattern, so the word can be found again) and that the alignment it reports matches the
// address it read.
#include "mtl_rt.h"

#include <cstdio>
#include <vector>

struct PtrArgs { uint32_t n; };

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"vk_ptr.slang"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline p = cx.pipeline(lib, "vk_ptr");
    const uint32_t n = 64;
    mtl::Buffer out = cx.alloc(n * 8);
    int bad = 0;
    for (uint32_t off = 0; off < 8; off++) {
        mtl::Buffer b = cx.alloc(n * 4);
        for (uint32_t i = 0; i < n * 4; i++) b.as<uint8_t>()[i] = uint8_t(0xA0 + i);
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(b, off), mtl::Arg::B(out), mtl::Arg::V(PtrArgs{n})};
        d.grid[0] = 1;
        d.tg[0] = 64;
        cx.run({d});
        const uint align = out.as<uint32_t>()[0], word = out.as<uint32_t>()[1];
        // a buffer's device address is 16-byte aligned (minStorageBufferOffsetAlignment), so the argument's
        // alignment is the offset's, and the word at or below it is the buffer's word at the offset below that
        const uint8_t* bytes = b.as<uint8_t>();
        const uint32_t at = off & ~3u;
        uint32_t want = 0;
        for (uint32_t i = 0; i < 4; i++) want |= uint32_t(bytes[at + i]) << (8 * i);
        const bool ok = word == want && align == (off & 3u);
        if (!ok) {
            std::printf("  offset %u: align %u (want %u) word %08x (want %08x)\n", off, align, off & 3u, word, want);
            bad++;
        }
    }
    std::printf("pointer->uint and the aligned word: %s (%d wrong)\n", bad ? "FAIL" : "ok", bad);
    return bad != 0;
}
