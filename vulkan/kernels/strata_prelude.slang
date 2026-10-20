// vulkan/kernels/strata_prelude.slang - included first in every library (Context::compile).
//
// Metal chooses the threadgroup size per dispatch; Vulkan fixes it in the pipeline, so the runtime makes one pipeline
// per size and passes it as specialization constants 100-102. Every kernel declares [numthreads(TG_X, TG_Y, TG_Z)].
// Metal's dynamic threadgroup memory (setThreadgroupMemoryLength at index 0) is TG_WORDS 4-byte words (constant 103).
[vk::constant_id(100)] const int TG_X = 32;
[vk::constant_id(101)] const int TG_Y = 1;
[vk::constant_id(102)] const int TG_Z = 1;
[vk::constant_id(103)] const int TG_WORDS = 1;

// The simdgroup is a 32-lane subgroup on Apple GPUs.
static const int SIMD = 32;
