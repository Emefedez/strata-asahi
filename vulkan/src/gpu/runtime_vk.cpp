// vulkan/src/gpu/runtime_vk.cpp - metal/include/strata/gpu/runtime.hpp for the Vulkan build (Asahi).
//
// On the Mac the shared host code's CUDA names (metal/include/metal_compat/cuda_runtime.h) reach metal/src/gpu/runtime.cpp.
// No kernel is launched through that layer: the engine's decode and prompt kernels go through mtl_rt.h
// (metal/src/engine/window.cpp), and the device copies the host code makes for the CUDA-shaped kernels are never read
// by the GPU (window.cpp gives their memory back, purge_device_allocations).  So on Vulkan this layer is host memory:
//   - an allocation is an anonymous mapping, and its "device pointer" is its CPU address (unified memory; host_view is
//     the identity on a known allocation);
//   - streams, events and graphs keep CUDA's ordering trivially: every copy and fill runs when it is issued, and a
//     capture records them to replay at graph_launch;
//   - purge_device_allocations drops the pages and makes the range inaccessible, so a late read faults at once instead
//     of reading zeros.
#include "strata/gpu/runtime.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace strata::gpu {

namespace {

struct Alloc {
    size_t bytes = 0;
    bool host = false;     // from host_alloc / host_register (the host code holds the CPU address)
    bool owned = true;     // false: host_register (memory the caller owns)
    bool purged = false;
};

struct Op {
    enum Kind { Copy, Fill } kind = Copy;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    int value = 0;
};

struct Runtime {
    std::mutex mu;
    std::map<uintptr_t, Alloc> by_addr;
};
Runtime& rt() {
    static Runtime* r = new Runtime;
    return *r;
}

thread_local Error t_last = kSuccess;
Error fail(Error e) {
    if (e != kSuccess && e != kNotReady) t_last = e;
    return e;
}

const size_t kPage = (size_t)sysconf(_SC_PAGESIZE);

/// The allocation holding `p`, or nullptr (ordinary host memory).
Alloc* find(const void* p, uintptr_t* base) {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.mu);
    const uintptr_t a = (uintptr_t)p;
    auto it = r.by_addr.upper_bound(a);
    if (it == r.by_addr.begin()) return nullptr;
    --it;
    if (a >= it->first + it->second.bytes) return nullptr;
    if (base) *base = it->first;
    return &it->second;
}

void check_purged(const void* p, const char* what) {
    if (Alloc* a = find(p, nullptr); a && a->purged) {
        std::fprintf(stderr, "strata vulkan runtime: a %s touches a device allocation of %.1f MB that was given back\n", what,
                     a->bytes / 1e6);
        std::abort();
    }
}

void alloc_log(const char* what, size_t n) {
    static const bool on = std::getenv("STRATA_METAL_ALLOC_LOG") != nullptr;
    static size_t tot = 0;
    tot += n;
    if (on && n >= (16u << 20)) std::fprintf(stderr, "strata vulkan runtime: %s %.1f MB (total %.2f GB)\n", what, n / 1e6, tot / 1e9);
}

Error new_alloc(void** p, size_t bytes, bool host) {
    const size_t n = (std::max<size_t>(bytes, 1) + kPage - 1) / kPage * kPage;
    void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (m == MAP_FAILED) return fail(kErrorMemoryAllocation);
    alloc_log(host ? "host alloc" : "device alloc", n);
    {
        std::lock_guard<std::mutex> lk(rt().mu);
        rt().by_addr[(uintptr_t)m] = Alloc{n, host, true, false};
    }
    *p = m;
    return kSuccess;
}

Error free_alloc(void* p) {
    if (!p) return kSuccess;
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.by_addr.find((uintptr_t)p);
    if (it == r.by_addr.end()) return fail(kErrorInvalidValue);
    if (it->second.owned) munmap(p, it->second.bytes);
    r.by_addr.erase(it);
    return kSuccess;
}

Error do_copy(void* dst, const void* src, size_t n) {
    check_purged(dst, "copy");
    check_purged(src, "copy");
    std::memmove(dst, src, n);
    return kSuccess;
}
Error do_fill(void* p, int v, size_t n) {
    check_purged(p, "fill");
    std::memset(p, v, n);
    return kSuccess;
}

}  // namespace

struct EventImpl {};
struct GraphImpl {
    std::vector<Op> ops;
};
struct GraphExecImpl {
    std::vector<Op> ops;
};
struct StreamImpl {
    GraphImpl* rec = nullptr;
};

namespace {
StreamImpl* legacy() {
    static StreamImpl* s = new StreamImpl;
    return s;
}
StreamImpl* S(Stream s) { return s ? s : legacy(); }
Error run_op(const Op& o) { return o.kind == Op::Copy ? do_copy(o.dst, o.src, o.bytes) : do_fill(o.dst, o.value, o.bytes); }
}  // namespace

// ---------------------------------------------------------------------------------------------------- runtime.hpp
void* host_view(const void* p) { return find(p, nullptr) ? const_cast<void*>(p) : nullptr; }

// window.cpp: the device copies of the weights the host code uploaded for the CUDA-shaped kernels are never read here;
// every device (not host) allocation of min_bytes or more gives its memory back.  Returns the bytes given back.
uint64_t purge_device_allocations(size_t min_bytes) {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.mu);
    uint64_t n = 0;
    for (auto& [addr, a] : r.by_addr) {
        if (a.host || a.purged || a.bytes < min_bytes) continue;
        madvise((void*)addr, a.bytes, MADV_DONTNEED);
        mprotect((void*)addr, a.bytes, PROT_NONE);
        a.purged = true;
        n += a.bytes;
    }
    return n;
}

Error get_last_error() { const Error e = t_last; t_last = kSuccess; return e; }
Error peek_last_error() { return t_last; }
const char* error_string(Error e) {
    switch (e) {
        case kSuccess: return "no error";
        case kNotReady: return "device not ready";
        case kErrorInvalidValue: return "invalid argument";
        case kErrorMemoryAllocation: return "out of memory";
        case kErrorUnsupported: return "operation not supported on Vulkan";
        case kErrorLaunchFailure: return "command buffer failed";
    }
    return "unknown runtime error";
}

Error get_device(int* dev) { *dev = 0; return kSuccess; }
Error set_device(int dev) { return dev == 0 ? kSuccess : fail(kErrorInvalidValue); }
Error device_count(int* n) { *n = 1; return kSuccess; }
Error device_synchronize() { return kSuccess; }
Error mem_get_info(size_t* free_b, size_t* total_b) {
    // unified memory: the machine's RAM, and what Linux can give without swapping (MemAvailable)
    *total_b = (size_t)sysconf(_SC_PHYS_PAGES) * kPage;
    *free_b = 0;
    std::ifstream f("/proc/meminfo");
    std::string k;
    size_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit)
        if (k == "MemAvailable:") { *free_b = v * 1024; break; }
    return kSuccess;
}
Error device_integrated(int* v, int) { *v = 1; return kSuccess; }
Error device_sm_count(int* v, int) { *v = 0; return fail(kErrorUnsupported); }
Error device_clock_khz(int* v, int) { *v = 0; return fail(kErrorUnsupported); }
Error runtime_version(int* v) { *v = 0; return kSuccess; }
Error device_can_access_peer(int* v, int, int) { *v = 0; return kSuccess; }
Error device_enable_peer_access(int, unsigned) { return fail(kErrorUnsupported); }
Error get_device_properties(DeviceProp* p, int dev) {
    if (dev != 0) return fail(kErrorInvalidValue);
    *p = DeviceProp{};
    std::snprintf(p->name, sizeof p->name, "Apple GPU (Vulkan, Honeykrisp)");
    std::snprintf(p->gcnArchName, sizeof p->gcnArchName, "apple9-or-older");
    p->warpSize = 32;
    size_t f = 0;
    mem_get_info(&f, &p->totalGlobalMem);
    return kSuccess;
}

Error malloc_device_raw(void** p, size_t bytes) { return new_alloc(p, bytes, false); }
Error free_device(void* p) { return free_alloc(p); }
Error host_alloc_raw(void** p, size_t bytes, unsigned) { return new_alloc(p, bytes, true); }
Error free_host(void* p) { return free_alloc(p); }
Error host_register(void* p, size_t bytes, unsigned) {
    std::lock_guard<std::mutex> lk(rt().mu);
    rt().by_addr[(uintptr_t)p] = Alloc{bytes, true, false, false};
    return kSuccess;
}
Error host_unregister(void* p) { return free_alloc(p); }
Error host_device_pointer_raw(void** dev, void* host) {
    if (!find(host, nullptr)) return fail(kErrorInvalidValue);
    *dev = host;
    return kSuccess;
}
Error memset_device(void* p, int value, size_t bytes) { return do_fill(p, value, bytes); }
Error memset_async(void* p, int value, size_t bytes, Stream st) {
    if (StreamImpl* s = S(st); s->rec) { s->rec->ops.push_back({Op::Fill, p, nullptr, bytes, value}); return kSuccess; }
    return do_fill(p, value, bytes);
}
Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind) { return do_copy(dst, src, bytes); }
Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind, Stream st) {
    if (StreamImpl* s = S(st); s->rec) { s->rec->ops.push_back({Op::Copy, dst, src, bytes, 0}); return kSuccess; }
    return do_copy(dst, src, bytes);
}
Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, CopyKind k, Stream st) {
    for (size_t r = 0; r < height; ++r) {
        const Error e = memcpy_async((uint8_t*)dst + r * dpitch, (const uint8_t*)src + r * spitch, width, k, st);
        if (e != kSuccess) return e;
    }
    return kSuccess;
}
Error memcpy_peer_async(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, Stream st) {
    if (dst_dev != 0 || src_dev != 0) return fail(kErrorInvalidValue);
    return memcpy_async(dst, src, bytes, kDeviceToDevice, st);
}

Error stream_create(Stream* s, unsigned) { *s = new StreamImpl; return kSuccess; }
Error stream_destroy(Stream s) { delete s; return kSuccess; }
Error stream_synchronize(Stream) { return kSuccess; }
Error stream_query(Stream) { return kSuccess; }
Error stream_wait_event(Stream, Event, unsigned) { return kSuccess; }
Error launch_host_func(Stream st, HostFn fn, void* arg) {
    if (S(st)->rec) return fail(kErrorUnsupported);
    fn(arg);
    return kSuccess;
}

Error event_create(Event* e, unsigned) { *e = new EventImpl; return kSuccess; }
Error event_destroy(Event e) { delete e; return kSuccess; }
Error event_record(Event, Stream) { return kSuccess; }
Error event_query(Event) { return kSuccess; }
Error event_synchronize(Event) { return kSuccess; }
Error event_elapsed_ms(float* ms, Event, Event) { *ms = 0.0f; return fail(kErrorUnsupported); }

Error begin_capture(Stream st) {
    StreamImpl* s = S(st);
    if (s->rec) return fail(kErrorInvalidValue);
    s->rec = new GraphImpl;
    return kSuccess;
}
Error end_capture(Stream st, Graph* g) {
    StreamImpl* s = S(st);
    if (!s->rec) { *g = nullptr; return fail(kErrorInvalidValue); }
    *g = s->rec;
    s->rec = nullptr;
    return kSuccess;
}
Error graph_node_count(Graph g, size_t* n) { *n = g ? g->ops.size() : 0; return kSuccess; }
Error graph_instantiate(GraphExec* x, Graph g) { *x = new GraphExecImpl{g->ops}; return kSuccess; }
Error graph_launch(GraphExec x, Stream st) {
    if (StreamImpl* s = S(st); s->rec) {
        s->rec->ops.insert(s->rec->ops.end(), x->ops.begin(), x->ops.end());
        return kSuccess;
    }
    for (const Op& o : x->ops)
        if (const Error e = run_op(o); e != kSuccess) return e;
    return kSuccess;
}
Error graph_upload(GraphExec, Stream) { return kSuccess; }
Error graph_destroy(Graph g) { delete g; return kSuccess; }
Error graph_exec_destroy(GraphExec x) { delete x; return kSuccess; }

}  // namespace strata::gpu
