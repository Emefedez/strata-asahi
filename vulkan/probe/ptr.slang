struct P { uint8_t* b; float* y; };
[[vk::push_constant]] P pc;
float rdh(uint8_t* p) { return float(*(half*)p); }
uint rd16(uint8_t* p) { return uint(*(uint16_t*)p); }
uint rd32(uint8_t* p) { uint16_t* s = (uint16_t*)p; return uint(s[0]) | (uint(s[1]) << 16); }
[shader("compute")] [numthreads(32,1,1)]
void k(uint3 g : SV_DispatchThreadID) {
    uint8_t* b = pc.b + 18 * g.x;
    float4 v = loadAligned<16>((float4*)(pc.y + 4));
    pc.y[g.x] = rdh(b) + float(rd16(b + 2)) + float(rd32(b + 6)) + float(b[3]) + v.x + WaveReadLaneAt(1.0f, WaveGetLaneIndex() ^ 1u);
}
