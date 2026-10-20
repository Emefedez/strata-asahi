// vulkan/tests/kcompile.cpp - compile a kernel library as the engine does and link entry points from it.
//   kcompile <file.metal,file2.metal,...> [--no-ggml] <entry> [entry...]
// Stops at the first Slang error (mtl_rt_vk prints it).
#include "mtl_rt.h"

#include <cstdio>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: kcompile <files,...> [--no-ggml] <entry>...\n"); return 2; }
    std::vector<std::string> files;
    std::stringstream ss(argv[1]);
    for (std::string f; std::getline(ss, f, ',');) files.push_back(f);
    int i = 2;
    std::string prelude;
    if (std::string(argv[i]) == "--no-ggml") { prelude = "#define NO_GGML_COMMON 1\n"; ++i; }
    mtl::Context cx;
    const double t0 = mtl::now_s();
    const int lib = cx.compile(files, prelude);
    std::printf("compiled %s in %.2f s\n", argv[1], mtl::now_s() - t0);
    for (; i < argc; ++i) {
        const double t1 = mtl::now_s();
        cx.pipeline(lib, argv[i]);
        std::printf("  %s ok (%.2f s)\n", argv[i], mtl::now_s() - t1);
    }
    return 0;
}
