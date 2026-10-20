// vulkan/kernels/vk_ptr.slang - can Slang turn a pointer into an integer?  The weight decoders need an address's
// alignment when a field has to be read with a 4-byte load (the blocks of some formats start at 2-byte aligned
// addresses).  The kernel reports the alignment of its buffer argument and the 4-byte word at or below it.
struct PtrArgs { uint n; };
struct PtrPush {
    uint8_t* p;
    uint* out;
    PtrArgs a;
};
[[vk::push_constant]] PtrPush pc;
[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void vk_ptr(uint3 gid : SV_DispatchThreadID) {
    if (gid.x >= pc.a.n) return;
    const uint64_t addr = (uint64_t)pc.p;
    const uint align = uint(addr & 3u);
    uint8_t* q = (uint8_t*)(addr - align);           // the 4-byte aligned word at or below the buffer
    const uint w = *(uint*)q;
    pc.out[2 * gid.x] = align;
    pc.out[2 * gid.x + 1] = w;
}
