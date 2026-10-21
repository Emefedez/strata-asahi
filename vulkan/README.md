# Strata on Asahi Linux (Vulkan)

A Vulkan engine for Strata on Apple Silicon Macs running Linux ([Asahi](https://asahilinux.org)). Upstream Strata
has CUDA, HIP, SYCL and Metal engines. Metal does not exist on Linux, and the upstream issue for Vulkan (#1214) was
closed as not planned. This directory is the port. Everything outside `vulkan/` is upstream Strata, plus a few
small hooks the port needs (see `git diff origin/main...vulkan-asahi`).

It runs Qwen3.8-Flash-Next (125B mixture of experts) on a 32 GB M1 Max. The dense weights and the most-used experts
stay in memory, and the rest stream from the SSD, the same design as the Mac engine.

## How it works

- `src/mtl_rt_vk.cpp` implements `metal/src/mtl_rt.h` (the Metal engine's runtime API) on Vulkan: buffers in
  unified memory, buffer device addresses for kernel arguments, push constants in the Metal argument layout, one
  pipeline per threadgroup size.
- `kernels/` holds the Metal kernels rewritten in [Slang](https://shader-slang.org). The decode window is complete:
  quant, GEMV, MoE, router, GDN, QSA, hyper-connections, sampling and MTP. Every prompt-chunk kernel is ported too
  (`dense_mm`, `moe_mm`, `prompt`, `gdn_prompt`, `qsa_prompt`), and each passes its parity test. The engine's prompt
  path has not been run end to end on Vulkan yet, so run configs still set `STRATA_METAL_PROMPT_GEMM=0` (prompts
  through decode windows).
- The Metal parity tests (`metal/tests/*_parity.cpp`, CPU references as the oracle) build against the Vulkan runtime
  unchanged (`-DSTRATA_VK_TESTS=gemv_parity;...`).
- Each part of a streamed window is its own submission, so the GPU is free while the host reads experts from the SSD.
  `STRATA_VK_SPIN=1` puts the whole window in one command buffer, with the GPU spinning on a host flag at each cut.
  That is faster (about 6 us a cut instead of 85-115 us), but it holds the GPU, and on 2026-10-21 it froze the
  desktop while a compile ran.

## Requirements

- An Apple Silicon Mac on Asahi Linux with 32 GB or more. It was tested only on the M1 Max 32 GB.
- [honeykrisp-strata](https://github.com/Emefedez/honeykrisp-strata)'s private Vulkan driver build. Stock Mesa caps
  the GPU heap at 50% of RAM and has no bf16 cooperative matrices. The private build is used per process
  (`VK_DRIVER_FILES`), and the desktop keeps its Mesa.
- Slang 2026.18.2 (the release tarball; the `shader-slang` package conflicts with S-Lang's `libslang.so`), cmake,
  ninja, and llama.cpp's ggml at the commit Strata pins (`setup.get_llama_cpp()`).
- About 40 GB of disk for the Q2_0 model, plus its streaming copy.

## Build and run

```sh
cmake -S vulkan -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_VULKAN=ON \
  -DSTRATA_GGML_DIR=third_party/llama.cpp -DSLANG_DIR=$HOME/.local/share/slang/2026.18.2
cmake --build build-vulkan --target strata

# download, pack, the experts' streaming copy, the MTP draft layer, the run config strata-<model>.json
.venv/bin/python vulkan/setup_asahi.py --model Q2_0 --no-start
```

[lmdeck-de](https://github.com/Emefedez/lmdeck-de) does all of this from its Backends screen and serves the
result next to its MLX and GGUF models.

Environment:

| Variable | Effect |
|---|---|
| `STRATA_VK_ARENA_BYTES` | cap on the expert arena (lmdeck sets it from free memory) |
| `STRATA_VK_SPIN=1` | one command buffer per window, the GPU spinning at each cut (see above) |
| `STRATA_VK_SPV_CACHE` | SPIR-V cache directory (default `~/.cache/strata-vulkan`, `0` = off) |
| `STRATA_VK_TIMING` | per-part GPU and host times |
| `STRATA_METAL_PROMPT_GEMM=0` | prompts through decode windows (required until the prompt path is wired in) |

## Measured

M1 Max 32 GB, Q2_0, experts streamed, 8 GB arena, greedy decoding. The commands and the rest are in
[RESULTS.md](RESULTS.md).

| Prompt | No drafts | Recommended settings (`--spec 8 --suffix-draft 3 --mtp ... --mtp-max-t 2 --spec-sched cost`) |
|---|---|---|
| A story (28 tokens) | 12.9 tok/s | 13.3 tok/s |
| A code rewrite (132 tokens) | 9.5 tok/s | 11.5 tok/s |

Prompts are read at ~10 tok/s because they still go through decode windows. Upstream's figure for a 32 GB Mac under
macOS (docs/MAC.md) is 7.9 tok/s of decode, but with a different prompt, so the two are not directly comparable.

`--spec-sched cost` (in `src/spec/draft_policy.cpp`) is an opt-in addition. It verifies an MTP draft only when the
calibrated chance of acceptance pays for the measured cost of the wider window. The idea comes from DeepSeek's
DeepInfer.

## Not done yet

- An end-to-end run of the prompt path (`STRATA_METAL_PROMPT_GEMM` unset) on the real model: its kernels pass parity,
  but the chunked path has not been timed or checked against the decode-window path on Asahi. `metal/tests/prompt_chain`
  cannot be used for that on a 32 GB machine: it wraps the whole 37 GB GGUF, which on Vulkan becomes a GPU copy.
- The dense GEMV reaches 163 GB/s, while its weight reads alone reach 322 GB/s. Its loads and arithmetic do not
  overlap yet.
- Vision input, and any Mac other than the M1 Max, are untested.
