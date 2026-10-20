// vulkan/kernels/bw_q2.slang - the Q2_0 weight reads of the GEMV, one pattern per MODE: GB/s of weight bytes.
//
// MODE 0: the codes as one 4-byte read at +2 of the block (2-byte aligned) and the scale as a 2-byte read
//         (what the kernel does today)
// MODE 1: the same codes, the scale as the 4-byte word that holds it
// MODE 2: the codes as two 4-byte words (aligned) and the scale as a 4-byte word
// MODE 3: the codes as one 16-byte read (the chunk that holds the unit) and the scale as a 4-byte word
// MODE 4: the codes as two 4-byte words (aligned), no scale
// MODE 5: 16-byte reads of the whole block (32 bytes fetched for the block's 18 useful bytes)
struct BwArgs { uint n_units; uint n_threads; uint mode; };
struct BwPush {
    uint8_t* src;
    uint* dst;
    BwArgs a;
};
[[vk::push_constant]] BwPush pc;

[shader("compute")]
[numthreads(TG_X, TG_Y, TG_Z)]
void bw_q2(uint3 gid : SV_DispatchThreadID) {
    const uint tid = gid.x;
    const uint stride = pc.a.n_threads;
    uint acc = 0u;
    for (uint i = tid; i < pc.a.n_units; i += stride) {
        const uint blk = i >> 2, j = i & 3u;
        uint8_t* b = pc.src + blk * 18u;
        const uint64_t ba = (uint64_t)b;
        const uint boff = uint(ba & 3u);
        uint code = 0u;
        if (pc.a.mode == 0) {   // the kernel today
            const uint16_t2 ac = loadAligned<2>((uint16_t2*)(b + 2 + 4 * j));
            code = uint(ac.x) | (uint(ac.y) << 16);
            acc ^= asuint(float(*(half*)b));
        } else if (pc.a.mode == 1) {   // the scale as an aligned word
            const uint16_t2 ac = loadAligned<2>((uint16_t2*)(b + 2 + 4 * j));
            code = uint(ac.x) | (uint(ac.y) << 16);
            acc ^= (loadAligned<4>((uint*)(ba - boff)) >> (boff * 8u)) & 0xFFFFu;
        } else if (pc.a.mode == 2) {   // the codes as two aligned words
            const uint64_t ca = ba + 2u + 4u * j;
            const uint coff = uint(ca & 3u);
            const uint lo = loadAligned<4>((uint*)(ca - coff));
            const uint hi = loadAligned<4>((uint*)(ca - coff + 4u));
            code = uint((uint64_t(lo) >> (coff * 8u)) | (uint64_t(hi) << (32u - coff * 8u)));
            acc ^= (loadAligned<4>((uint*)(ba - boff)) >> (boff * 8u)) & 0xFFFFu;
        } else if (pc.a.mode == 3) {   // the codes as one 16-byte read
            const uint64_t ca = ba + 2u + 4u * j;
            const uint4 v = loadAligned<16>((uint4*)(ca & ~15ull));
            code = v.x ^ v.y ^ v.z ^ v.w;
            acc ^= (loadAligned<4>((uint*)(ba - boff)) >> (boff * 8u)) & 0xFFFFu;
        } else if (pc.a.mode == 4) {   // the codes as two aligned words, no scale
            const uint64_t ca = ba + 2u + 4u * j;
            const uint coff = uint(ca & 3u);
            const uint lo = loadAligned<4>((uint*)(ca - coff));
            const uint hi = loadAligned<4>((uint*)(ca - coff + 4u));
            code = uint((uint64_t(lo) >> (coff * 8u)) | (uint64_t(hi) << (32u - coff * 8u)));
        } else {   // two 16-byte reads of the block
            const uint4 v0 = loadAligned<16>((uint4*)(ba & ~15ull));
            const uint4 v1 = loadAligned<16>((uint4*)((ba & ~15ull) + 16u));
            code = v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
        }
        acc ^= code;
    }
    pc.dst[tid] = acc;
}
