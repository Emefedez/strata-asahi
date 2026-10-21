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

## The engine end to end (2026-10-21)

`build-vulkan/strata` (`cmake -S vulkan -B build-vulkan -DSTRATA_ENABLE_VULKAN=ON -DSTRATA_GGML_DIR=<llama.cpp at the
pinned commit>`), Q2_0, the experts streamed from the SSD (`STRATA_METAL_STREAM=1`), the prompt through decode windows
(`STRATA_METAL_PROMPT_GEMM=0`: the prompt kernels are not ported yet), greedy.  A short question: "The capital of
France is Paris."  A story prompt (28 tokens), 120 tokens generated:

| | arena | decode |
|---|---|---|
| no MTP | 4 GB | 8.56 tok/s (hit rate 0.71, 283 MB from the SSD a token) |
| no MTP | 8 GB | 10.49 tok/s (0.84, 201 MB) |
| no MTP, the handshake's fence polled every 4096 spins (not every spin) | 8 GB | **11.63 tok/s** |
| `--mtp`, `--spec 2` | 8 GB | 9.12 tok/s (68% of drafts accepted, 1.67 tokens a round) |
| `--mtp`, `--spec 2 --spec-sched cost` | 8 GB | 10.03 tok/s (87.5% of the verified drafts accepted) |
| `--mtp`, `--spec 3 --spec-sched cost` | 8 GB | 8.30 tok/s |

The tokens are the same in every run.  Upstream's Mac number for this M1 Max under macOS (docs/MAC.md) is 7.9 tok/s;
the prompts differ, so it is not a like-for-like comparison yet.  The least MemAvailable in any run: 8.2 GB.

### Where a token's time goes (STRATA_VK_TIMING, STRATA_METAL_KPROF=T with STRATA_METAL_KPROF_STREAM=1)

| a window of | GPU between the 48 handshakes | the host's turns (SSD) | tail (head, sample) |
|---|---|---|---|
| 1 token | 39.5 ms | 22-25 ms | 3.1 ms |
| 2 tokens | 63.5 ms | 37-62 ms | 5.0 ms |

The GPU time of a 1-token window is 42 ms for ~3.5 GB of weights (~83 GB/s): the experts 11 ms (moe_gu 7.0, moe_down
4.0), the hyper-connection reads ~8 ms, the LM head 2.8 ms (Q5_K, 437 MB, 156 GB/s), GDN 2.4, QSA 1.8, ~40 small
dense GEMV classes the rest.  The draft round (MTP layer + draft head) is 3.0 ms of GPU and 0.07 ms of encoding.

**Why speculation does not pay yet:** the second token of a window costs 60% of a token on the GPU and more on the
SSD.  The dense GEMVs (the lane kernels of gemv.slang) grow almost linearly with the window: their cost is the
activation loads (64 bytes per 16 weights, per token) and the per-token arithmetic, not the weights.  `mma_parity
--bench` (the LM head, Q5_K): lane 2.61 / 6.36 / 13.55 ms for 1 / 3 / 8 tokens; the matrix-unit GEMV (mma.slang,
8x8x8 cooperative matrices) 12.8 ms flat - flat as it should be, but 34 GB/s.  `STRATA_METAL_GEMV_NR=2` (two rows a
lane share each x load) does not change the 2-token window's GPU time.  MoE grows only 25% for twice the experts (at
one token it leaves the GPU idle).

### What to do next, for speculation

1. A matrix-unit GEMV that is fast at 1-4 tokens (the weights decoded once, the window on the 8x8 tiles): today's
   mma.slang decodes in one simdgroup per 8 rows with wave barriers between; at 34 GB/s it only breaks even at 8
   tokens.  Using it for every window size keeps the outputs identical with and without MTP (it is bitwise the same
   for 1, 3 and 8 tokens, `mma_parity`), not identical to the lane GEMV's.
2. The hyper-connection reads (`gr_down_q8`, 1.6 -> 3.5 ms a class for the second token): the same x-traffic problem.
3. The SSD for the draft rows: a rejected draft's experts are read too (+14-37 ms of host turns at 2 tokens).
4. `--spec-sched cost` (DraftPolicy::choose_mtp): works as designed (a draft is verified when its calibrated
   acceptance pays for the measured cost of the wider window; costs measured in the first, cold rounds are
   re-measured after 40 rounds); it will pay as soon as 1-3 make the wider window cheap.

### Measurement artifacts to know

The first window of each size compiles its pipeline specialisations (0.5-1.7 s on Asahi); `STRATA_METAL_PROFILE`'s
averages and `mtp ... ms/round drafting` include it (the "41 ms of patch+finish" and the "15-24 ms a draft round" of
the first profiles were this).  `STRATA_METAL_SKIP` changes the routing (empty kernels feed the router garbage), so
its tok/s mixes the GPU time with the SSD's; the per-window GPU time is the number to read.

## Speculation, with the kernels compiled once (2026-10-21, after the SPIR-V cache)

The first window of every size used to compile ~60 kernels (2-3 s, Slang's code generation per specialization),
which had made every wider window look slow.  With one SPIR-V module per entry point and the disk cache, warm runs
(Q2_0, experts streamed, arena 8 GB, greedy, 150 tokens; the tokens are the same in every row of a table):

A code rewrite (the prompt asks for a 14-line Python function back with type hints; 132 prompt tokens):

| | tokens a round | decode |
|---|---|---|
| no drafts | 1.00 | 9.50 tok/s |
| `--spec 8 --suffix-draft 3` (prompt lookup) | 2.14 (91% of drafts accepted) | 11.19 tok/s |
| `--spec 8 --suffix-draft 3 --mtp <dir> --mtp-max-t 2 --spec-sched cost` | 3.43 (97%) | **11.54 tok/s (+21%)** |

The story prompt (28 tokens):

| | tokens a round | decode |
|---|---|---|
| no drafts | 1.00 | 12.94 tok/s |
| `--spec 8 --suffix-draft 3` | 1.00 (nothing to look up) | 12.59 tok/s |
| `--spec 8 --suffix-draft 3 --mtp <dir> --mtp-max-t 2 --spec-sched cost` | 1.32 (80%) | **13.34 tok/s** |

The prompt through decode windows went from 7.3 to 9.9 tok/s with the cache (the code prompt's time to the first
token 22.9 -> 15.8 s).  The recommended decode settings for a 32 GB Mac on Asahi: the last row of both tables.
