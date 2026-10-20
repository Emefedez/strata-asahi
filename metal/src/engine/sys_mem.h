// metal/src/engine/sys_mem.h - what the engine asks the OS about memory, on macOS and on Linux (the Vulkan build on
// Asahi compiles these engine files too).
//   phys_ram()        the RAM (hw.memsize / sysconf)
//   proc_mem()        what the process holds: macOS phys_footprint and resident size; Linux RssAnon + RssShmem + VmSwap
//                     (the GPU buffers are shmem-backed GEM objects) and VmRSS
//   mem_available()   what the OS can give without paging anything out: macOS free + purgeable + the clean file cache
//                     (arena_avail), Linux MemAvailable
//   filecache_min()   macOS's file-cache floor (vm.vm_page_filecache_min); Linux has none (0)
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#include "arena_budget.h"

namespace strata::metal_engine {

struct ProcMem {
    double footprint = 0, rss = 0;   // bytes; 0 when unknown
};

inline uint64_t phys_ram() {
#if defined(__APPLE__)
    uint64_t phys = 0;
    size_t len = sizeof phys;
    sysctlbyname("hw.memsize", &phys, &len, nullptr, 0);
    return phys;
#else
    return (uint64_t) sysconf(_SC_PHYS_PAGES) * (uint64_t) sysconf(_SC_PAGESIZE);
#endif
}

#if !defined(__APPLE__)
/// The kB value of `key` in a /proc file of "Key:  value kB" lines, in bytes; -1 when absent.
inline double proc_kb(const char* path, const char* key) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    char line[256];
    const size_t kl = std::strlen(key);
    double v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, key, kl) == 0 && line[kl] == ':') {
            unsigned long long kb = 0;
            if (std::sscanf(line + kl + 1, "%llu", &kb) == 1) v = (double) kb * 1024.0;
            break;
        }
    std::fclose(f);
    return v;
}
#endif

inline ProcMem proc_mem() {
    ProcMem r;
#if defined(__APPLE__)
    task_vm_info_data_t ti;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &ti, &cnt) == KERN_SUCCESS) {
        r.footprint = (double) ti.phys_footprint;
        r.rss = (double) ti.resident_size;
    }
#else
    const double anon = proc_kb("/proc/self/status", "RssAnon"), shm = proc_kb("/proc/self/status", "RssShmem"),
                 swap = proc_kb("/proc/self/status", "VmSwap"), rss = proc_kb("/proc/self/status", "VmRSS");
    if (anon >= 0) r.footprint = anon + (shm > 0 ? shm : 0) + (swap > 0 ? swap : 0);
    if (rss >= 0) r.rss = rss;
#endif
    return r;
}

/// `fallback` when the OS does not say.
inline double mem_available(double fallback) {
#if defined(__APPLE__)
    vm_statistics64_data_t vs;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &cnt) == KERN_SUCCESS) {
        const double pg = (double) vm_page_size;
        return arena_avail(vs.free_count * pg, vs.purgeable_count * pg, vs.inactive_count * pg, vs.speculative_count * pg,
                           vs.external_page_count * pg);
    }
    return fallback;
#else
    const double a = proc_kb("/proc/meminfo", "MemAvailable");
    return a >= 0 ? a : fallback;
#endif
}

inline double filecache_min() {
#if defined(__APPLE__)
    unsigned long long pages = 0;   // a 4-byte count of pages, read into a zeroed 8
    size_t pl = sizeof pages;
    if (sysctlbyname("vm.vm_page_filecache_min", &pages, &pl, nullptr, 0) == 0) return (double) pages * (double) vm_page_size;
#endif
    return 0;
}

}  // namespace strata::metal_engine
