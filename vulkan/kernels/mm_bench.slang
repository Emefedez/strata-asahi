// vulkan/kernels/mm_bench.slang - the coop_layout probe of metal/kernels/mm_bench.metal (tests/coop_probe.h) only.  On
// the Mac it reports the element layout Metal 4 gives matmul2d's right-input cooperative tensor (the M5's runs of 4 k);
// the Vulkan kernels have no such tensor and take the M1-M4 paths, so every lane reports a layout that is not the M5's
// (cap 4, elements (0, 0)): coop_k4() answers false.
struct ProbePush { int* out; };
void coop_fill(ProbePush pc, uint lane) {
    int* q = pc.out + lane * 128 * 4;
    for (int i = 0; i < 128 * 4; ++i) q[i] = 0;
    q[2] = 4;
}
[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void coop_layout_16_64_32(uint tid : SV_GroupIndex, uniform ProbePush pc) { coop_fill(pc, tid); }
[shader("compute")] [numthreads(TG_X, TG_Y, TG_Z)]
void coop_layout_16_32_32(uint tid : SV_GroupIndex, uniform ProbePush pc) { coop_fill(pc, tid); }
