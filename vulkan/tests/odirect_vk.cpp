// vulkan/tests/odirect_vk.cpp - can O_DIRECT reads land in Vulkan-mapped memory (the streaming arena)?
//
//   odirect_vk <file> [MiB=1024] [chunk KiB=1792]
// Reads `MiB` of `file` in `chunk`-sized O_DIRECT preads, 8 threads, into (a) a Vulkan buffer's mapping and (b) plain
// anonymous memory, and prints GB/s for each.  A driver whose mapping is VM_PFNMAP fails (a) with EFAULT.
#include "mtl_rt.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double run(int fd, uint8_t* dst, size_t total, size_t chunk, off_t base, int* err) {
    std::atomic<size_t> next{0};
    std::atomic<int> e{0};
    const size_t n = total / chunk;
    std::vector<std::thread> th;
    const double t0 = mtl::now_s();
    for (int t = 0; t < 8; ++t)
        th.emplace_back([&] {
            for (size_t i; (i = next++) < n;) {
                const ssize_t r = pread(fd, dst + i * chunk, chunk, base + (off_t)(i * chunk));
                if (r != (ssize_t)chunk) { e = r < 0 ? errno : -1; return; }
            }
        });
    for (auto& x : th) x.join();
    *err = e;
    return total / (mtl::now_s() - t0) / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: odirect_vk <file> [MiB] [chunk KiB]\n"); return 2; }
    const size_t mib = argc > 2 ? std::atoi(argv[2]) : 1024, chunk = (argc > 3 ? std::atoi(argv[3]) : 1792) * 1024ull;
    const size_t total = mib * 1024 * 1024 / chunk * chunk;
    const int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) { std::perror("open"); return 2; }
    mtl::Context cx;
    mtl::Buffer b = cx.alloc(total);
    void* anon = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    std::memset(anon, 0, total);
    int e = 0;
    // different file ranges for each so the page cache (O_DIRECT bypasses it anyway) cannot help
    const int reps = 4;
    for (int rep = 0; rep < reps; ++rep) {   // alternating, a new file range each time
        const off_t o = (off_t)(rep + 1) * (off_t)(2 * total);   // past range 0, which earlier runs read
        const double g2 = run(fd, (uint8_t*)anon, total, chunk, o + (off_t)total, &e);
        std::printf("anonymous      : %s %.2f GB/s\n", e ? std::strerror(e > 0 ? e : EIO) : "ok", e ? 0.0 : g2);
        const double g1 = run(fd, b.as<uint8_t>(), total, chunk, o, &e);
        std::printf("vulkan mapping : %s %.2f GB/s\n", e ? std::strerror(e > 0 ? e : EIO) : "ok", e ? 0.0 : g1);
    }
    // the data must be the same as a buffered read
    std::vector<uint8_t> ref(chunk);
    const int fb = open(argv[1], O_RDONLY);
    if (pread(fb, ref.data(), chunk, (off_t)reps * (off_t)(2 * total)) != (ssize_t)chunk) return 3;
    std::printf("content check  : %s\n", std::memcmp(ref.data(), b.ptr, chunk) == 0 ? "equal" : "DIFFERENT");
    return 0;
}
