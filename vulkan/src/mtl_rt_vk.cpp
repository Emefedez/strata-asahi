// vulkan/src/mtl_rt_vk.cpp - metal/src/mtl_rt.h on Vulkan, for Apple GPUs under Linux (Honeykrisp, Asahi).
//
// The engine's kernel host code (metal/src/engine) and the parity programs (metal/tests) talk to the GPU only through
// mtl_rt.h; this file gives them the same calls on Vulkan, so they build unchanged:
//   * kernels: compile() takes the .metal names the callers pass and compiles the Slang ports of the same files
//     (vulkan/kernels/<name>.slang, same entry-point names and function-constant indices) at run time, as Metal does;
//     pipeline() picks an entry point; the Vulkan pipeline itself is made per threadgroup size at the first dispatch
//     (Vulkan fixes the workgroup size in the pipeline: specialization constants 100-102, the dynamic threadgroup
//     memory 103, in 4-byte words);
//   * arguments: every dispatch's arguments go into push constants in C layout - a buffer argument is its GPU
//     address (buffer device address) + offset, inline bytes keep their alignment - and the kernels declare the same
//     struct ([[vk::push_constant]], scalar layout). The Metal kernels already take device pointers this way;
//   * memory: one host-visible, coherent allocation per buffer (unified memory: Buffer::ptr is the CPU view);
//   * a serial run() puts a barrier after every dispatch (Metal's serial encoder), run_graph() before the dispatches
//     marked `barrier`, and run_graph_seg() waits for the host inside the command buffer (handshake.comp).
// Times are wall clock (submit to fence): Honeykrisp's compute timestamps are not usable (they read 28,000 GB/s).
#include "mtl_rt.h"
#include "resource_path.h"

#include <slang-com-ptr.h>
#include <slang.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "handshake_spv.h"   // kHandshakeSpv: handshake.comp, compiled at build time

#ifndef STRATA_VULKAN_KERNELS_DEFAULT
#define STRATA_VULKAN_KERNELS_DEFAULT "vulkan/kernels"
#endif

namespace mtl {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "mtl: cannot read %s\n", path.c_str());
        std::exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

#define VK(x)                                                                                    \
    do {                                                                                         \
        VkResult r_ = (x);                                                                       \
        if (r_ != VK_SUCCESS) {                                                                  \
            std::fprintf(stderr, "mtl(vulkan): %s failed (%d) at %s:%d\n", #x, r_, __FILE__, __LINE__); \
            std::exit(5);                                                                        \
        }                                                                                        \
    } while (0)

namespace {

constexpr uint32_t kPushBytes = 256;      // maxPushConstantsSize on Honeykrisp
constexpr uint32_t kSpecWgX = 100, kSpecWgY = 101, kSpecWgZ = 102, kSpecTgWords = 103;

struct BufImpl {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceAddress addr = 0;
    size_t bytes = 0;
};

struct PipeImpl {
    std::string fn;
    VkShaderModule mod = VK_NULL_HANDLE;
    std::vector<Constants::V> consts;
    std::mutex mu;
    std::map<std::array<uint32_t, 4>, VkPipeline> by_shape;   // (tg x, y, z, tg memory words)
};

BufImpl* BI(const Buffer& b) { return static_cast<BufImpl*>(b.handle); }
size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

struct Context::Impl {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t mem_type = 0;
    VkPhysicalDeviceProperties props{};
    VkDeviceSize heap_size = 0;
    std::string name;
    Slang::ComPtr<slang::IGlobalSession> gs;
    Slang::ComPtr<slang::ISession> session;
    std::vector<Slang::ComPtr<slang::IModule>> libs;
    // run_graph_seg's handshake
    PipeImpl* hs = nullptr;
    Buffer flags;
    uint32_t seq = 0;

    VkPipeline pipeline_for(PipeImpl* p, const uint32_t tg[3], uint32_t tg_mem) {
        const uint32_t words = std::max<uint32_t>(1, (tg_mem + 3) / 4);
        const std::array<uint32_t, 4> key{tg[0], tg[1], tg[2], words};
        std::lock_guard<std::mutex> g(p->mu);
        auto it = p->by_shape.find(key);
        if (it != p->by_shape.end()) return it->second;
        std::vector<VkSpecializationMapEntry> entries;
        std::vector<uint32_t> data;
        auto add = [&](uint32_t id, uint32_t v) {
            entries.push_back({id, (uint32_t)(data.size() * 4), 4});
            data.push_back(v);
        };
        for (const auto& c : p->consts) add((uint32_t)c.index, c.type == 1 ? (c.v ? VK_TRUE : VK_FALSE) : c.v);
        add(kSpecWgX, tg[0]);
        add(kSpecWgY, tg[1]);
        add(kSpecWgZ, tg[2]);
        add(kSpecTgWords, words);
        VkSpecializationInfo si{(uint32_t)entries.size(), entries.data(), data.size() * 4, data.data()};
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, p->mod,
                    p->fn.c_str(), &si};
        ci.layout = layout;
        VkPipeline pl;
        VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pl));
        p->by_shape.emplace(key, pl);
        return pl;
    }

    static void barrier(VkCommandBuffer cb) {
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void encode(VkCommandBuffer cb, const Dispatch& d) {
        require_fit(d, "dispatch");
        auto* p = static_cast<PipeImpl*>(d.p->handle);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_for(p, d.tg, d.tg_mem));
        uint8_t pc[kPushBytes] = {};
        size_t o = 0;
        for (const Arg& a : d.args) {
            if (a.buf) {
                o = align_up(o, 8);
                const uint64_t addr = BI(*a.buf)->addr + a.off;
                if (o + 8 > kPushBytes) goto too_big;
                std::memcpy(pc + o, &addr, 8);
                o += 8;
            } else {
                o = align_up(o, std::max<uint32_t>(4, a.align));
                if (o + a.bytes.size() > kPushBytes) goto too_big;
                std::memcpy(pc + o, a.bytes.data(), a.bytes.size());
                o += a.bytes.size();
            }
        }
        o = align_up(o, 4);
        if (o) vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)o, pc);
        if (d.indirect)
            vkCmdDispatchIndirect(cb, BI(*d.indirect)->buf, d.ind_off);
        else
            vkCmdDispatch(cb, d.grid[0], d.grid[1], d.grid[2]);
        return;
    too_big:
        std::fprintf(stderr, "mtl(vulkan): %s: arguments exceed %u bytes of push constants\n", d.p->name.c_str(), kPushBytes);
        std::exit(5);
    }

    VkCommandBuffer begin() {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VK(vkAllocateCommandBuffers(dev, &ai, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(cb, &bi));
        return cb;
    }

    VkFence submit(VkCommandBuffer cb) {
        VK(vkEndCommandBuffer(cb));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence f;
        VK(vkCreateFence(dev, &fi, nullptr, &f));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VK(vkQueueSubmit(q, 1, &si, f));
        return f;
    }

    void finish(VkCommandBuffer cb, VkFence f) {
        const VkResult r = vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "mtl(vulkan): command buffer failed (%d)\n", r);
            std::exit(5);
        }
        vkDestroyFence(dev, f, nullptr);
        vkFreeCommandBuffers(dev, pool, 1, &cb);
    }
};

static std::string g_device_name = "unknown GPU";

void require_fit(const Pipeline& p, uint32_t threads, const char* where) {
    if ((int)threads > p.max_threads) {
        std::fprintf(stderr, "mtl: %s: %s needs %u threads per threadgroup, the GPU gives %d\n", where, p.name.c_str(),
                     threads, p.max_threads);
        std::exit(6);
    }
}

Context::Context() : impl_(new Impl) {
    Impl& m = *impl_;
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "strata";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VK(vkCreateInstance(&ici, nullptr, &m.inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(m.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(m.inst, &n, pds.data());
    for (auto p : pds) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(p, &pr);
        if (pr.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            m.pd = p;
            m.props = pr;
            break;
        }
    }
    if (!m.pd) {
        std::fprintf(stderr, "mtl(vulkan): no Vulkan GPU\n");
        std::exit(2);
    }
    m.name = m.props.deviceName;
    g_device_name = m.name;
    vkGetPhysicalDeviceQueueFamilyProperties(m.pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(n);
    vkGetPhysicalDeviceQueueFamilyProperties(m.pd, &n, qfs.data());
    for (uint32_t i = 0; i < n; i++)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m.qf = i;
            break;
        }

    // Features the kernels use; each is checked, the device refuses unsupported ones.
    VkPhysicalDeviceVulkan11Features have11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &have11};
    VkPhysicalDeviceVulkan13Features have13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &have12};
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &have13};
    vkGetPhysicalDeviceFeatures2(m.pd, &have);
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.storageBuffer16BitAccess = have11.storageBuffer16BitAccess;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.vulkanMemoryModel = VK_TRUE;
    f12.vulkanMemoryModelDeviceScope = have12.vulkanMemoryModelDeviceScope;
    f12.scalarBlockLayout = VK_TRUE;
    f12.shaderInt8 = have12.shaderInt8;
    f12.shaderFloat16 = have12.shaderFloat16;
    f12.storageBuffer8BitAccess = have12.storageBuffer8BitAccess;
    f12.shaderSubgroupExtendedTypes = have12.shaderSubgroupExtendedTypes;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    f13.synchronization2 = VK_TRUE;
    f13.maintenance4 = VK_TRUE;   // LocalSizeId: the workgroup size as specialization constants
    f13.subgroupSizeControl = have13.subgroupSizeControl;
    f13.computeFullSubgroups = have13.computeFullSubgroups;
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    f.features.shaderInt64 = have.features.shaderInt64;
    f.features.shaderInt16 = have.features.shaderInt16;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m.qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK(vkCreateDevice(m.pd, &dci, nullptr, &m.dev));
    vkGetDeviceQueue(m.dev, m.qf, 0, &m.q);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpi.queueFamilyIndex = m.qf;
    VK(vkCreateCommandPool(m.dev, &cpi, nullptr, &m.pool));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK(vkCreatePipelineLayout(m.dev, &pli, nullptr, &m.layout));

    // Host-visible coherent memory, device-local and CPU-cached when offered (unified memory: one pool).
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m.pd, &mp);
    int best = -1, best_score = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const auto fl = mp.memoryTypes[i].propertyFlags;
        if (!(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || !(fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) continue;
        const int score = ((fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 2 : 0) + ((fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0);
        if (score > best_score) best = (int)i, best_score = score;
    }
    if (best < 0) {
        std::fprintf(stderr, "mtl(vulkan): no host-visible coherent memory type\n");
        std::exit(2);
    }
    m.mem_type = (uint32_t)best;
    m.heap_size = mp.memoryHeaps[mp.memoryTypes[best].heapIndex].size;

    kernel_dir = strata_metal::resource("STRATA_VULKAN_KERNELS", "vulkan-kernels", STRATA_VULKAN_KERNELS_DEFAULT);

    if (SLANG_FAILED(slang::createGlobalSession(m.gs.writeRef()))) {
        std::fprintf(stderr, "mtl(vulkan): Slang did not start\n");
        std::exit(2);
    }
    slang::TargetDesc td{};
    td.format = SLANG_SPIRV;
    td.profile = m.gs->findProfile("spirv_1_6");
    std::vector<slang::CompilerOptionEntry> opts;
    auto opt_int = [&](slang::CompilerOptionName name, int v) {
        slang::CompilerOptionEntry e{};
        e.name = name;
        e.value.kind = slang::CompilerOptionValueKind::Int;
        e.value.intValue0 = v;
        opts.push_back(e);
    };
    opt_int(slang::CompilerOptionName::GLSLForceScalarLayout, 1);
    opt_int(slang::CompilerOptionName::VulkanUseEntryPointName, 1);
    opt_int(slang::CompilerOptionName::Optimization, SLANG_OPTIMIZATION_LEVEL_HIGH);
    td.compilerOptionEntries = opts.data();
    td.compilerOptionEntryCount = (uint32_t)opts.size();
    slang::SessionDesc sd{};
    sd.targets = &td;
    sd.targetCount = 1;
    const char* search[] = {kernel_dir.c_str()};
    sd.searchPaths = search;
    sd.searchPathCount = 1;
    if (SLANG_FAILED(m.gs->createSession(sd, m.session.writeRef()))) {
        std::fprintf(stderr, "mtl(vulkan): Slang session failed\n");
        std::exit(2);
    }

    // The handshake kernel (precompiled SPIR-V) and its flag words.
    m.hs = new PipeImpl;
    m.hs->fn = "main";
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(kHandshakeSpv);
    smi.pCode = kHandshakeSpv;
    VK(vkCreateShaderModule(m.dev, &smi, nullptr, &m.hs->mod));
    m.flags = alloc(256);
}

Context::~Context() {
    if (impl_->dev) vkDeviceWaitIdle(impl_->dev);
    delete impl_;
}

std::string Context::device_name() const { return impl_->name; }

// Apple GPU families by chip: M1 = Apple7, M2 = Apple8, M3 = Apple9 (Honeykrisp names the device "Apple M1 Max ...").
bool Context::has_family_apple(int n) const {
    int fam = 7;
    if (impl_->name.find("M2") != std::string::npos) fam = 8;
    if (impl_->name.find("M3") != std::string::npos) fam = 9;
    return n <= fam;
}
bool Context::supports_metal4() const { return false; }
uint64_t Context::recommended_working_set() const { return impl_->heap_size; }
uint64_t Context::max_buffer_length() const { return impl_->heap_size; }

int Context::compile(const std::vector<std::string>& files, const std::string& prelude, bool) {
    // The callers name the .metal files of the Metal tree; the Slang ports carry the same names (an absolute path,
    // the engine's own kernels, maps to kernel_dir/engine/<name>).
    std::string src = prelude + "\n#include \"strata_prelude.slang\"\n";
    if (prelude.find("NO_GGML_COMMON") == std::string::npos) src += "#include \"ggml_common.slang\"\n";
    std::string last;
    for (const auto& f : files) {
        std::string base = f.substr(f.rfind('/') == std::string::npos ? 0 : f.rfind('/') + 1);
        if (base.size() > 6 && base.compare(base.size() - 6, 6, ".metal") == 0) base = base.substr(0, base.size() - 6) + ".slang";
        const std::string p = (f.size() && f[0] == '/') ? kernel_dir + "/engine/" + base : kernel_dir + "/" + base;
        src += "#line 1 \"" + base + "\"\n" + read_file(p) + "\n";
        last = base;
    }
    const std::string name = "lib" + std::to_string(impl_->libs.size());
    Slang::ComPtr<slang::IBlob> diag;
    slang::IModule* mod = impl_->session->loadModuleFromSourceString(name.c_str(), (name + ".slang").c_str(), src.c_str(), diag.writeRef());
    if (!mod) {
        std::fprintf(stderr, "mtl(vulkan): compile failed (%s; %s):\n%s\n", impl_->name.c_str(), last.c_str(),
                     diag ? (const char*)diag->getBufferPointer() : "");
        std::exit(3);
    }
    impl_->libs.push_back(Slang::ComPtr<slang::IModule>(mod));
    return (int)impl_->libs.size() - 1;
}

Pipeline Context::pipeline(int lib, const std::string& fn, const Constants& c) {
    slang::IModule* mod = impl_->libs.at(lib).get();
    Slang::ComPtr<slang::IEntryPoint> ep;
    if (SLANG_FAILED(mod->findEntryPointByName(fn.c_str(), ep.writeRef()))) {
        std::fprintf(stderr, "mtl(vulkan): no function %s\n", fn.c_str());
        std::exit(3);
    }
    slang::IComponentType* parts[] = {mod, ep.get()};
    Slang::ComPtr<slang::IComponentType> composite, linked;
    Slang::ComPtr<slang::IBlob> diag, code;
    if (SLANG_FAILED(impl_->session->createCompositeComponentType(parts, 2, composite.writeRef(), diag.writeRef())) ||
        SLANG_FAILED(composite->link(linked.writeRef(), diag.writeRef())) ||
        SLANG_FAILED(linked->getEntryPointCode(0, 0, code.writeRef(), diag.writeRef()))) {
        std::fprintf(stderr, "mtl(vulkan): pipeline %s:\n%s\n", fn.c_str(), diag ? (const char*)diag->getBufferPointer() : "");
        std::exit(3);
    }
    auto* p = new PipeImpl;
    p->fn = fn;
    p->consts = c.vals;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code->getBufferSize();
    smi.pCode = static_cast<const uint32_t*>(code->getBufferPointer());
    VK(vkCreateShaderModule(impl_->dev, &smi, nullptr, &p->mod));
    Pipeline r;
    r.handle = p;
    r.max_threads = (int)impl_->props.limits.maxComputeWorkGroupInvocations;
    if (const char* cap = std::getenv("STRATA_METAL_MAX_THREADS_CAP")) {
        const int lim = std::atoi(cap);
        if (lim > 0 && lim < r.max_threads) r.max_threads = lim;
    }
    r.simd_width = 32;
    r.static_tg_mem = 0;
    r.name = fn;
    if (!c.vals.empty()) {
        r.name += "{";
        for (size_t i = 0; i < c.vals.size(); ++i)
            r.name += (i ? "," : "") + std::to_string(c.vals[i].index) + "=" + std::to_string((int)c.vals[i].v);
        r.name += "}";
    }
    return r;
}

Buffer Context::alloc(size_t bytes) {
    auto* b = new BufImpl;
    b->bytes = std::max<size_t>(bytes, 16);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = b->bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    VK(vkCreateBuffer(impl_->dev, &bci, nullptr, &b->buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(impl_->dev, b->buf, &mr);
    VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fl};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = impl_->mem_type;
    const VkResult r = vkAllocateMemory(impl_->dev, &mai, nullptr, &b->mem);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "mtl(vulkan): alloc %zu bytes failed (%d)\n", bytes, r);
        std::exit(4);
    }
    VK(vkBindBufferMemory(impl_->dev, b->buf, b->mem, 0));
    void* ptr = nullptr;
    VK(vkMapMemory(impl_->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &ptr));
    std::memset(ptr, 0, b->bytes);
    VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    ai.buffer = b->buf;
    b->addr = vkGetBufferDeviceAddress(impl_->dev, &ai);
    allocated_ += b->bytes;
    Buffer out;
    out.handle = b;
    out.ptr = ptr;
    out.bytes = bytes;
    return out;
}

// Honeykrisp cannot import host pages (no VK_EXT_external_memory_host; the Asahi GPU driver has no user-pointer
// import), so a wrap is a GPU allocation filled with a copy. Buffer::ptr is then the allocation's CPU view, not `p`:
// callers that keep writing to `p` afterwards must write to the buffer instead (the Vulkan engine's arena does).
Buffer Context::wrap(void* p, size_t bytes) {
    Buffer b = alloc(bytes);
    std::memcpy(b.ptr, p, bytes);
    return b;
}
Buffer Context::wrap(void* p, size_t bytes, size_t) { return wrap(p, bytes); }

void Context::free(Buffer& b) {
    if (auto* bi = BI(b)) {
        vkDeviceWaitIdle(impl_->dev);
        vkDestroyBuffer(impl_->dev, bi->buf, nullptr);
        vkFreeMemory(impl_->dev, bi->mem, nullptr);
        allocated_ -= std::min<uint64_t>(allocated_, bi->bytes);
        delete bi;
    }
    b = Buffer{};
}

double Context::run(const std::vector<Dispatch>& ds, int reps, bool concurrent, bool barrier_between) {
    VkCommandBuffer cb = impl_->begin();
    for (int r = 0; r < reps; ++r)
        for (size_t i = 0; i < ds.size(); ++i) {
            impl_->encode(cb, ds[i]);
            if (!concurrent || barrier_between) Impl::barrier(cb);   // a serial encoder orders every dispatch
        }
    const double t0 = now_s();
    VkFence f = impl_->submit(cb);
    impl_->finish(cb, f);
    return now_s() - t0;
}

double Context::run_graph(const std::vector<Dispatch>& ds) {
    VkCommandBuffer cb = impl_->begin();
    for (size_t i = 0; i < ds.size(); ++i) {
        if (i > 0 && ds[i].barrier) Impl::barrier(cb);
        impl_->encode(cb, ds[i]);
    }
    const double t0 = now_s();
    VkFence f = impl_->submit(cb);
    impl_->finish(cb, f);
    return now_s() - t0;
}

double Context::run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, void (*at_cut)(void*, int),
                              void* user) {
    Impl& m = *impl_;
    auto* fl = static_cast<uint32_t*>(m.flags.ptr);
    const uint32_t base = m.seq;
    // ~4 s of polling before a wait gives up; the host's turn (SSD reads for one layer) takes milliseconds
    const uint32_t max_iter = 100u * 1000u * 1000u;
    VkCommandBuffer cb = m.begin();
    size_t i = 0;
    for (size_t s = 0; s <= cuts.size(); ++s) {
        const size_t end = s < cuts.size() ? (size_t)cuts[s] : ds.size();
        const size_t first = i;
        for (; i < end; ++i) {
            if (i > first && ds[i].barrier) Impl::barrier(cb);
            m.encode(cb, ds[i]);
        }
        if (s < cuts.size()) {
            Impl::barrier(cb);
            const uint32_t tg[3] = {1, 1, 1};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m.pipeline_for(m.hs, tg, 0));
            struct { uint64_t flags; uint32_t seq, max_iter; } pc{BI(m.flags)->addr, base + (uint32_t)s + 1, max_iter};
            vkCmdPushConstants(cb, m.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, 1, 1, 1);
            Impl::barrier(cb);
        }
    }
    const double t0 = now_s();
    VkFence f = m.submit(cb);
    for (size_t s = 0; s < cuts.size(); ++s) {
        const uint32_t want = base + (uint32_t)s + 1;
        while (__atomic_load_n(&fl[0], __ATOMIC_ACQUIRE) < want) {
            if (vkGetFenceStatus(m.dev, f) == VK_SUCCESS) break;   // finished early (a timeout): reported below
        }
        at_cut(user, (int)s);
        __atomic_store_n(&fl[1], want, __ATOMIC_RELEASE);
    }
    m.finish(cb, f);
    m.seq = base + (uint32_t)cuts.size();
    if (const uint32_t to = __atomic_exchange_n(&fl[2], 0u, __ATOMIC_ACQ_REL)) {
        std::fprintf(stderr, "mtl(vulkan): %u host turns took longer than the GPU waits for (results are not valid)\n", to);
        std::exit(5);
    }
    return now_s() - t0;
}

double Context::run_wall(const std::vector<Dispatch>& ds, int reps, double* gpu_seconds) {
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) run(ds, 1, false, false);
    const double dt = now_s() - t0;
    if (gpu_seconds) *gpu_seconds = dt;
    return dt;
}

double Context::run_icb(const std::vector<Dispatch>& ds, int reps, bool barrier_between) {
    return run(ds, reps, !barrier_between, barrier_between);
}

void Context::warm(const std::vector<Dispatch>& ds, double seconds) {
    const double t0 = now_s();
    while (now_s() - t0 < seconds) run(ds, 1, false, false);
}

}  // namespace mtl
