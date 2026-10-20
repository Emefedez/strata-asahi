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

// Metal's simd_shuffle_xor / simd_prefix_exclusive_sum on the 32-lane subgroup.
float shfl_xor(float v, uint m) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ m); }
int shfl_xor(int v, uint m) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ m); }
uint shfl_xor(uint v, uint m) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ m); }
int simd_prefix_exclusive_sum(int v) { return WavePrefixSum(v); }

// The flat threadgroup size (Metal's threads_per_threadgroup for a 1-D group).
static const int TG_N = TG_X * TG_Y * TG_Z;
static const float FLT_MAX = 3.402823466e+38f;
#define INFINITY asfloat(0x7f800000u)
