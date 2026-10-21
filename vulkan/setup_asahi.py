#!/usr/bin/env python3
"""vulkan/setup_asahi.py - set up Strata on an Apple Silicon Mac running Linux (Asahi): the Vulkan engine.

The Mac's setup (metal/setup_mac.py) with the Linux parts in its place. Its steps 5-7 are the same - the model
download, the pack (tools/iq_pack.py), the experts' streaming copy (metal/tools/stream_experts.py) and the run config
- because the Vulkan engine is the Metal engine's design: unified memory, the experts streamed from the SSD on a
32 GB machine. What differs:
  1. the machine check: RAM from /proc/meminfo, the Apple GPU through Vulkan, the driver (honeykrisp-strata's
     private ICD: the device heap above 50% of RAM);
  4. the engine: vulkan/ (build-vulkan/strata) when this tree has it;
  and the model folder is marked no-copy-on-write on btrfs (chattr +C: no checksums or compression on the 37+ GB
  files; reads of 1.75 MB experts measured 4-7% faster without them on the M1 Max).

    python3 vulkan/setup_asahi.py --model Q2_0 --yes --no-start
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metal"))
import setup as S  # noqa: E402
import setup_mac as M  # noqa: E402

BUILD_DIR = ROOT / "build-vulkan"
EXE = BUILD_DIR / "strata"
DEFAULT_ICD = Path.home() / ".local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json"


def linux_info() -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        mem[k] = int(v.split()[0]) * 1024
    chip = "Apple Silicon"
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "deviceName" in line and "Apple" in line:
                chip = line.split("=", 1)[1].strip()
                break
    except (OSError, subprocess.TimeoutExpired):
        pass
    # mac_fit's thresholds are in decimal GB as macOS reports them (a 32 GiB Mac is "34 GB")
    return {"chip": chip, "ram_gb": mem["MemTotal"] / 1e9, "available_gb": mem.get("MemAvailable", 0) / 1e9,
            "macos": None}


def linux_problems(info: dict) -> list:
    out = []
    if "Apple" not in info["chip"]:
        out.append("no Apple GPU seen through Vulkan (vulkaninfo): this setup is for Apple Silicon under Asahi Linux")
    if info["ram_gb"] < 17:
        out.append(f"{info['ram_gb']:.0f} GB of memory: the model needs at least 16 GB (it streams from the SSD)")
    return out


def no_cow(folder: Path) -> None:
    """chattr +C on an empty btrfs folder: the files made in it skip copy-on-write, checksums and compression."""
    try:
        fs = subprocess.run(["stat", "-f", "-c", "%T", str(folder)], capture_output=True, text=True).stdout.strip()
        if fs == "btrfs" and not any(folder.iterdir()):
            subprocess.run(["chattr", "+C", str(folder)], check=False)
            S.ok(f"{folder}: no copy-on-write (btrfs)")
    except OSError:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=[m for m in M.MAC_MODELS], default=None)
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--data-dir")
    ap.add_argument("--move-data", action="store_true")
    ap.add_argument("--set-default-data-dir", action="store_true")
    ap.add_argument("--models-dir")
    ap.add_argument("--gguf-dir")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--source", choices=S.SOURCES, default=None)
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-mtp", action="store_true", help="no MTP draft layer (it is ~5 GB to download and 0.9 GB of memory)")
    a = ap.parse_args(argv)
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    S.say("Strata on Asahi Linux (Apple Silicon, Vulkan)")
    data, elsewhere = S.data_folder_for(a)
    roots = [data, *elsewhere]
    models_root = Path(a.models_dir) if a.models_dir else data / "models"

    # ---- 1. the machine
    S.step(1, "checking this Mac (Linux)")
    info = linux_info()
    S.ok(f"{info['chip']}, {info['ram_gb']:.0f} GB memory ({info['available_gb']:.0f} GB available now)")
    icd = Path(os.environ.get("VK_DRIVER_FILES") or DEFAULT_ICD)
    if icd.exists():
        S.ok(f"Vulkan driver for the engine: {icd}")
    else:
        S.warn(f"the engine's Vulkan driver ({icd}) is missing: lmdeck's `strata install` builds it")
    problems = linux_problems(info)
    for p in problems:
        S.warn(p)
    free = S.free_gb(data if data.exists() else ROOT)
    S.ok(f"free disk: {free:.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        S.fail(problems[0])

    # ---- 2. the choices (the Mac's rules: they are about unified memory and the SSD, as here)
    S.step(2, "your choices")
    rec, notes = M.recommend_model(info["ram_gb"])
    for n in notes:
        S.warn(n)
    model = a.model or rec
    ctx = a.context or M.recommend_context(info["ram_gb"])
    S.ok(f"{model}, context {ctx}")
    family = M.MAC_MODELS[model].get("family", "qwen")
    fam = S.FAMILIES[family]
    tag = fam["tag"] + model
    need = (0 if a.gguf_dir else S.MODELS[model]["download_gb"]) + 6
    if free < need:
        S.warn(f"about {need:.0f} GB are needed on the disk ({free:.0f} GB free): it may run out")

    # ---- 3. Python packages
    S.step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES, "numpy, jinja2, ...")

    # ---- 4. the engine
    S.step(4, "the Strata engine (Vulkan)")
    llama = S.get_llama_cpp()
    S.ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:7]} (ggml, gguf-py)")
    if EXE.exists():
        S.ok(f"engine: {EXE}")
    else:
        S.warn(f"the Vulkan engine is not built yet ({EXE}): the model is prepared and the run config written; it "
               "loads once the engine is built")

    # ---- 5. the model
    S.step(5, f"downloading {fam['title']} {model}")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else models_root / tag
    models_dir.mkdir(parents=True, exist_ok=True)
    no_cow(models_dir)
    files = [models_dir / n for n in M.model_shard_names(model)]
    for s in files:
        if s.exists() and S.done(s):
            S.ok(f"{s.name} already downloaded")
            continue
        S.download(fam["hf"].format(q=model) + s.name, s)
    S.check_shards(files)
    for s in files:
        if M.shard_sha(s.name):
            S.verify_sha256(s, *M.shard_sha(s.name))
    S.ok("model files present")

    # ---- 6. the pack and the experts' streaming copy
    S.step(6, "preparing the model for Strata")
    if len(files) > 2:
        S.fail("this setup prepares the two-shard packs (Q2_0, IQ3_XXS, IQ3_S); the Unsloth files are not set up here yet")
    pack = S.find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        S.run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(files[0]), "--out", str(pack),
               *fam.get("pack_args", [])], env=env)
    S.ok(f"model prepared: {pack}")
    stream = M.streams(model, info["ram_gb"])
    rt = data / "mtp" / "rt"
    if stream:
        S.ok("the experts stream from the SSD")
        M.experts_file(files[0])
    # The MTP draft layer, also when the experts stream: on Asahi a draft is verified only when its measured
    # acceptance pays for the wider window (--spec-sched cost; vulkan/RESULTS.md: +21% on a code rewrite, +3% on a
    # story, the same tokens)
    use_mtp = not a.no_mtp
    if use_mtp and not (rt.parent / "mtp-q2_0.gguf").exists():
        S.say("  The MTP draft layer (speculative decoding) comes from the original Qwen checkpoint: only its ~5 GB of MTP tensors are downloaded.")
        S.fetch_mtp(rt.parent, env)
        S.run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(rt.parent), "--experts", "q2_0",
               "--out", str(rt.parent / "mtp-q2_0.gguf")], env=env)
    if use_mtp:
        rt.mkdir(parents=True, exist_ok=True)   # the engine reads mtp-q2_0.gguf beside it
        S.refresh_draft_vocab(rt, "cjk")
        S.ok(f"MTP draft layer: {rt.parent / 'mtp-q2_0.gguf'}")

    # ---- 7. the run config
    S.step(7, "writing the run config")
    port = a.port or 8080
    cfg = M.mac_config(EXE, vk_args(M.engine_args(pack, files, rt, ctx, stream, info["ram_gb"]), rt, use_mtp), pack,
                       f"{fam['name']}-{model.lower()}", tag, port)
    cfg["backend"] = "vulkan"
    # STRATA_METAL_PROMPT_GEMM=0: prompts through the decode windows until the prompt kernels are ported to Vulkan
    cfg["env"] = {"STRATA_METAL_STREAM": "1" if stream else "0", "VK_DRIVER_FILES": str(icd), "STRATA_METAL_PROMPT_GEMM": "0"}
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    S.write_config(cfg_path, cfg)
    S.ok(f"run config: {cfg_path} (lmdeck lists it as strata/{tag.lower()})")
    return 0


def vk_args(args: list, rt: Path, use_mtp: bool) -> list:
    """The Mac's engine arguments with the decode settings measured on Asahi (vulkan/RESULTS.md): windows up to 8
    tokens for prompt-lookup drafts, the MTP's own windows at 2 tokens, scheduled by expected tokens per ms."""
    out, i = [], 0
    drop = {"--spec": 1, "--suffix-draft": 1, "--mtp": 1, "--mtp-max-t": 1, "--spec-sched": 1}
    while i < len(args):
        if args[i] in drop:
            i += 1 + drop[args[i]]
            continue
        out.append(args[i])
        i += 1
    out += ["--spec", "8", "--suffix-draft", "3"]
    if use_mtp:
        out += ["--mtp", str(rt), "--mtp-max-t", "2", "--spec-sched", "cost"]
    return out


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        S.say("\nstopped.")
