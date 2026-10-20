# vulkan/RESULTS.md - the Vulkan port's measurements, what they say, and what to do next

Everything here was measured on the M1 Max (32 GB, Asahi Linux 7.1.12, honeykrisp-strata v3, Mesa 26.3.0-devel
`f144c7949b`) on 2026-10-20. The driver is the private ICD of
`~/.local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json`; the engine's kernels are the Slang
ports in `vulkan/kernels/`.

## The commands

```sh
# the driver: what a load of each width reaches (honeykrisp-strata's bench)
cd ~/Work/honeykrisp-strata/bench && make
VK_DRIVER_FILES=$HOME/.local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json \
  ./build/vkprobe --mib 512 bw_width

# the Q2_0 weight reads on their own, one pattern at a time (this tree)
ninja -C build-vulkan
VK_DRIVER_FILES=$HOME/.local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json \
  ./build-vulkan/bw_q2 512 8

# the dense GEMV: every format's parity and the model's shapes
VK_DRIVER_FILES=$HOME/.local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json \
  ./build-vulkan/gemv_parity            # parity: 13 formats, 3 shapes, 1/3/8 tokens
VK_DRIVER_FILES=$HOME/.local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json \
  ./build-vulkan/gemv_parity --bench    # the model's shapes, the best tile each
```

## What the driver does with a load

`bench/bw_width` (honeykrisp-strata) reads 512 MiB with one load width and counts the *useful* bytes:

| the load | GB/s |
|---|---|
| 16 bytes (uvec4) | 288-292 |
| 8 bytes (uint2) | 274-288 |
| 4 bytes (uint) | 260-277 |
| 2 bytes (ushort) | **67-71** |

A 2-byte load reaches a quarter of the bandwidth. A 4-byte load is as good as a 16-byte one, and it may sit at any
2-byte aligned address: the Q2_0 pattern (a 4-byte read of the unit's codes at +2 of the block, then the block's 2-byte
scale) reads at **322 GB/s** - the fastest of the six patterns `vulkan/tests/bw_q2.cpp` measures.

So the rule for the port: **never read a weight with a 1-byte or 2-byte load, and do not bother with 16-byte loads.**
The earlier note ("move to 16-byte loads") was measuring the wrong thing: the 4-byte read was already there.

## What the Q2_0 GEMV spends its time on

`build-vulkan/gemv_parity --one Q2_0 16384 16384`: 16384 x 16384 Q2_0, one token, 75.5 MB of weights, **163 GB/s**
(463 us). The bench probes (function constant 9) split that up, in milliseconds per GB of weights:

| | ms/GB | GB/s |
|---|---|---|
| the whole kernel | 6.15 | 163 |
| without the weight loads and the decode (probe 5) | 1.94 | 516 |
| without the activation loads (probe 4) | 5.78 | 173 |
| **the weight path alone** (probe 5 subtracted) | **4.31** | **232** |

The weight path on its own reaches 322 GB/s (`build-vulkan/bw_q2`), so the kernel is **not** limited by the weight
bandwidth: its time is the *sum* of the weight loads (4.31), the activations (0.47) and the rest (1.94). Nothing
overlaps. With the weights read at the DRAM speed the ceiling of this kernel shape is about 186 GB/s; past that the
loads and the arithmetic have to overlap.

## What the port changed, and what it bought

The decoders read a block's 16-bit arrays as 4-byte pairs (`rd2x2` in `vulkan/kernels/quant.slang`) instead of one
16-bit word at a time. The values do not change: `gemv_parity` passes for all 13 formats (worst rel L2 5.3e-7, every
window bitwise equal to one-token calls). The speed, five runs a shape, the median (`gemv_parity --one`):

| shape | format | before | after |
|---|---|---|---|
| 640 x 2560 | Q4_0 | 78.8 GB/s | **82.5 GB/s** |
| 2560 x 10240 | Q8_0 | 265.6 GB/s | **277.7 GB/s** |
| 2560 x 10240 | Q3_K | 97.3 GB/s | **102.3 GB/s** |
| 16384 x 16384 | Q2_0 | 162.8 GB/s | 162.9 GB/s |
| 2560 x 10240 | Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S | 157-206 GB/s | unchanged |
| 10240 x 320 | BF16 | 229.1 GB/s | 230.8 GB/s |

The formats whose bulk was read a 16-bit word at a time gain 5%; the rest do not move. Two things that did *not* work,
and why:

- **Reading a misaligned field as two 4-byte words** (to keep every load 4-byte aligned) costs two loads and a 64-bit
  merge per field: the Q2_0 GEMV fell from 162 to 121 GB/s. A 4-byte read at a 2-byte aligned address is fine.
- **Reading the i-quant byte arrays as words** (IQ3_S) cost more in arithmetic than it saved: 92.7 -> 84.8 GB/s.
  Those bytes stay byte reads.

## What to do next

1. **Overlap the weight loads with the arithmetic** (the biggest lever left). The kernel loads a unit, decodes it and
   combines it, then loads the next one: the memory wait is exposed. Fetching the next unit's weights before decoding
   the current one (two units in flight) should take the kernel from 163 GB/s towards the 232 GB/s its weight path
   already reaches, and further if the loads and the decode overlap.
2. **Cheaper decodes.** The decode is about 20% of the Q2_0 kernel's time (92 us of 470 for 16384 x 16384). The Q2_0
   decode is already minimal (a mask per code); the K-quants' decoders are not.
3. **The MoE expert kernels** (not ported yet). They read far more weights per token than the dense GEMV (the
   experts are ~700 MB a token against the dense 75 MB), so their weight path matters more. The same rule applies:
   4-byte reads, no repacking. The staging through shared memory is only worth it if a kernel's weight path turns out
   to be bandwidth-bound rather than instruction-bound.
4. **The engine** (`build-vulkan/strata`) is not built yet: the model's pack and the experts' streaming copy are done
   (`vulkan/setup_asahi.py` finished on 2026-10-20), the kernels are the next piece, then the host code of
   `metal/src/engine` on Vulkan.

## The numbers that are not measurements

- The Mac's numbers (upstream's `docs/MAC.md`: 7.9 tok/s decode on a 32 GB M4 mini) are not comparable yet: there is
  no end-to-end run on Linux.
- The dense GEMV's ceiling of 186 GB/s is a sum of the probe's numbers, not a measurement.
