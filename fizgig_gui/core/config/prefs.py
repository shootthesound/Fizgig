import json
import os

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

# ---------------------------------------------------------------------------
# Preferences — centralized model and directory paths (Klein 9B only for now)
# ---------------------------------------------------------------------------

PREFS_FILE = os.path.join(_REPO_ROOT, "prefs.json")
HELP_FILE = os.path.join(_REPO_ROOT, "help.json")

# --- Captioning models -----------------------------------------------------
# Florence-2 downloads itself from HuggingFace on first use; the Qwen3-VL entry appears only when
# the Krea 2 text-encoder file is present in Preferences. That file is a full vision-language
# model with a real LM head, so it can genuinely describe an image — and the captions it writes
# are plain .txt, so it is useful for ANY dataset, Klein runs included. The label names where the
# file comes from, not who it is for.
FLORENCE_DEFAULT_MODEL = "MiaoshouAI/Florence-2-base-PromptGen"
FLORENCE_MODELS = [FLORENCE_DEFAULT_MODEL, "microsoft/Florence-2-base", "microsoft/Florence-2-large"]
# Florence-2 isn't a native transformers architecture, so loading it means trust_remote_code=True
# — downloading and EXECUTING whatever Python is currently on that repo's default branch, with no
# pin. Pinned here to the commit each was audited against, so a compromised account (or a repo
# that just changes later) can't silently change what gets executed on someone's next first-run.
# To refresh a pin: check https://huggingface.co/api/models/<repo> for the current "sha".
FLORENCE_REVISIONS = {
    "MiaoshouAI/Florence-2-base-PromptGen": "da7ac9f3deac56a928e2fd4d94d8bb985d231299",
    "microsoft/Florence-2-base": "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
    "microsoft/Florence-2-large": "21a599d414c4d928c9032694c424fb94458e3594",
}
# PromptGen's config doesn't carry its own modeling code — its auto_map points at
# "microsoft/Florence-2-base-ft--modeling_florence2...", so transformers fetches the code that
# actually EXECUTES from that second, different repo. Our `revision` above only pins PromptGen
# itself; transformers only carries it over to the code repo automatically when the two repos
# are the same one (they aren't here), so the redirected repo needs its own explicit pin via
# code_revision or it silently stays on "main". The two microsoft/ models don't redirect
# (their auto_map has no repo prefix, just the module path), so they don't need an entry here.
FLORENCE_CODE_REVISIONS = {
    "MiaoshouAI/Florence-2-base-PromptGen": "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",  # microsoft/Florence-2-base-ft
}
FLORENCE_TASKS = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
QWEN_CAPTION_MODEL = "Qwen3-VL 4B (Krea 2 text encoder)"
QWEN_CUSTOM_TASK = "Custom…"

# Prefs whose VALUE is a directory we want to keep portable across repo
# clones/moves. When saved to disk, paths inside _FIZGIG_DIR are stored as
# relative strings (with forward slashes); when loaded, they're resolved back
# to absolute so every consumer works unchanged. Paths outside the repo (e.g.
# a user pointing Cache to another drive) stay absolute on both sides.
#
# Model paths (base_dit, vae, text_encoder, ...) are NOT in this set — they
# point to external HuggingFace downloads and are always absolute.
#
# Note: `dataset_dir` is deliberately NOT a pref — the dataset TOML path is
# fully hardcoded to FIZGIG_DIR/dataset/ via the DATASET_DIR module constant
# (Dataset tab auto-saves to Fizgig_train.toml; no browse UI anywhere), so
# exposing it in Preferences was dead weight.
_PORTABLE_DIR_KEYS = {"lora_output_dir", "profiles_dir", "cache_dir"}


def _resolve_pref_path(value: str) -> str:
    """Convert a possibly-relative stored path to an absolute path by joining
    with _FIZGIG_DIR. Absolute paths are returned unchanged."""
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(_FIZGIG_DIR, value))


def _serialize_pref_path(value: str) -> str:
    """Store a path as relative-to-_FIZGIG_DIR if it lives inside the repo;
    otherwise store as an absolute path. Uses forward slashes for
    cross-platform portability of the JSON file."""
    if not value:
        return value
    try:
        abs_value = os.path.abspath(value)
        fizgig_abs = os.path.abspath(_FIZGIG_DIR)
        rel = os.path.relpath(abs_value, fizgig_abs)
    except ValueError:
        # Windows: path is on a different drive than the repo — relpath raises.
        return os.path.abspath(value).replace(os.sep, "/")
    if rel.startswith("..") or os.path.isabs(rel):
        # Outside the repo — keep absolute.
        return os.path.abspath(value).replace(os.sep, "/")
    return rel.replace(os.sep, "/")


DEFAULT_PREFS = {
    # Model paths (absolute — point to external model downloads).
    # Blank on first launch; user fills these in via the Preferences tab. Each
    # row has a "Download" link that opens the correct HuggingFace repo.
    "base_dit": "",
    "distilled_dit": "",
    "vae": "",
    "text_encoder": "",
    # Krea 2 model paths. RAW = training base; Turbo (pre-quant fp8) = previews/
    # inference; Qwen-Image VAE; Qwen3-VL-4B text encoder (bf16 for training).
    "krea2_raw_dit": "",
    "krea2_turbo_dit": "",
    "krea2_vae": "",
    "krea2_text_encoder": "",
    # Turbo distillation LoRA (rank 64) — RAW + this at strength 1.0 behaves as the Turbo
    # model, so samples can render on the resident training DiT instead of loading the
    # separate Turbo checkpoint (saves the park-to-CPU shuffle during previews).
    "krea2_turbo_lora": "",
    # MiniMax H3 model paths (experimental third family — barebones image-only LoRA training).
    # bf16 DiT is the training base (NF4-quantized at load); Qwen3-VL-32B TE + video VAE cache.
    "minimax_dit": "",
    # ref2va is a DIFFERENT fine-tune from fl2va, not another quantization of it: it is what
    # ComfyUI's r2v workflow loads, and the only H3 build that accepts reference images.
    # Optional — required only for reference distillation.
    "minimax_ref_dit": "",
    "minimax_text_encoder": "",
    "minimax_vae": "",
    # Audio VAE — optional, and only video clips ever use it. With it, a clip's sound becomes a
    # real training target; without it, clips train video only, which is what every dataset did
    # before clips existed. Never loaded for a stills folder.
    "minimax_audio_vae": "",
    # Turbo LoRA — optional, previews only: 6-step in-training samples with the community Turbo
    # applied at ~75% on top of the training adapter, exactly how fast ComfyUI inference runs it.
    "minimax_turbo_lora": "",
    # Output directories — relative to repo root, portable across clones/moves.
    # Resolved to absolute in load_prefs(); in-memory pref values are absolute.
    # All three live as top-level folders inside the repo:
    #   FizgigIndependent/output_loras/
    #   FizgigIndependent/profiles/
    #   FizgigIndependent/cache/
    # (Dataset TOMLs are always in FIZGIG_DIR/dataset/ via DATASET_DIR — not
    # a pref; see _PORTABLE_DIR_KEYS note above.)
    "lora_output_dir": "output_loras",
    "profiles_dir": "profiles",
    "cache_dir": "cache",
    # Input directories — default folders the Browse dialogs open in, so users
    # don't re-hunt every session. input_lora_dir seeds LoRA pickers (Repair
    # Studio primary/donor, LoRA the Explorer, Context LoRA on Training);
    # input_ref_dir seeds the reference-image pickers (Repair Studio, Explorer).
    # Absolute paths (may live anywhere); empty = no default (last folder).
    "input_lora_dir": "",
    "input_ref_dir": "",
    # Where the Start tab's training-folder Browse opens. On a pod the entrypoint seeds this to
    # /workspace/datasets so Browse lands where the uploads are, rather than at cwd.
    "input_dataset_dir": "",
    # Stop the RunPod pod when a training run finishes cleanly. Off by default: a rented GPU bills
    # by the hour, so a run that ends at 4am otherwise bills until someone notices — but stopping
    # a machine out from under someone has to be something they asked for.
    "runpod_stop_when_done": "0",
    # An account API key that can stop pods. Stored here rather than expected as a template env
    # var because a PUBLIC template hands its variables to everyone who deploys it — one person's
    # key would end up controlling their account from strangers' containers. Kept on the user's
    # own volume, entered in the RunPod card, masked in the UI.
    "runpod_api_key": "",
    # Inference DiT block swap — int 0-16 for Klein 9B. With the Distilled fp8
    # model (workbench default) 0 = no swap fits ~16GB; loading Base is heavier
    # (0 ≈ 24GB). 16 = max swap for the smallest cards. Applies to Repair Studio,
    # Profiler, and Extractor. The Training tab has its own separate BLOCKS_SWAP.
    "inference_blocks_to_swap": "Auto (detect from GPU)",
    # INT8 fast inference: quantize the workbench/preview DiT's block Linears to int8 (W8A8) for a
    # faster matmul. On by default — same VRAM as fp8 (8-bit either way), composes with block swap,
    # and only affects previews (never the saved LoRA). Toggle off in Preferences to use fp8.
    "inference_int8": "1",
    # Which physical GPU to use, as a bare index ("0", "1", ...). Empty = leave the machine's
    # default alone, which is what every install before this had, so single-GPU users see no
    # change. Applied by exporting CUDA_VISIBLE_DEVICES (see _apply_cuda_device_pref): torch
    # then renumbers the chosen card to cuda:0 and nothing downstream - trainer, loader,
    # sampler, cache scripts - has to know a choice was made at all.
    "cuda_device": "",
}


def _enumerate_gpus():
    """[(index, name, total_gb, uuid)] for every card in the machine, or [] if it cannot be read.

    Deliberately NOT torch.cuda: touching it creates the CUDA context, which fixes the visible
    device set for the life of the process - i.e. asking torch what GPUs exist would defeat the
    setting this list feeds. NVML and nvidia-smi both enumerate the real hardware regardless of
    CUDA_VISIBLE_DEVICES, which is exactly what a chooser needs. UUID is included so that
    CUDA_VISIBLE_DEVICES can be set by UUID — immune to NVML vs. CUDA index reordering on
    Windows when the display GPU is not the fastest card (issue #104)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            out.append((i, name.decode() if isinstance(name, bytes) else name,
                        pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3), uuid))
        if out:
            return out
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=6,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = []
        for line in r.stdout.strip().splitlines():
            idx, name, mb, uuid = [p.strip() for p in line.split(",")]
            out.append((int(idx), name, int(mb) / 1024, uuid))
        return out
    except Exception:
        return []


def _apply_cuda_device_pref(prefs) -> str:
    """Export CUDA_VISIBLE_DEVICES from the saved pref. Returns what was applied, or "".

    This is the whole GPU-selection mechanism: everything downstream keeps asking for "cuda"
    and gets the chosen card, because it is the only one it can see. Must run before anything
    creates a CUDA context - the variable is read once, when that context is built.

    An existing CUDA_VISIBLE_DEVICES in the environment wins: someone who launched with it set
    meant it, and silently overriding that from a saved pref would be worse than not having the
    pref at all."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    want = str(prefs.get("cuda_device", "")).strip()
    if want:
        os.environ["CUDA_VISIBLE_DEVICES"] = want
        return want
    return ""


def _auto_detect_blocks_to_swap() -> int:
    """Pick a DiT block-swap preset based on the GPU's total VRAM.

    Only called when the user hasn't saved an explicit preference yet.
    Returns the leading integer for the swap-preset labels (0/4/8/12/16).
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # 16 GB and up → no swap. The Distilled fp8 model (workbench default) is
            # light, so 16 GB cards run previews without swapping, and skipping swap
            # avoids the PCIe latency. Threshold is 15, not 16: a 16 GB card reports
            # ~15.9 GiB total (drivers reserve a little), so a >=16 gate would wrongly
            # exclude it. Only genuinely smaller cards (<15 GB) swap.
            if vram_gb >= 15:
                return 0   # 16 GB / 24 GB / 32 GB — no swap needed
            if vram_gb >= 10:
                return 12  # 12-14 GB
            return 16      # <10 GB — maximum swap
    except Exception:
        pass
    return 0  # safe fallback


def load_prefs() -> dict:
    """Load user preferences from prefs.json, falling back to defaults.
    Relative portable-dir paths are resolved to absolute for in-memory use."""
    prefs = dict(DEFAULT_PREFS)
    user_set_swap = False
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                user_set_swap = "inference_blocks_to_swap" in saved
                prefs.update(saved)
        except Exception:
            pass
    # Inference block swap defaults to "Auto (detect from GPU)", resolved at each
    # pipeline load (see _get_inference_blocks_to_swap) — same behaviour as the
    # training Blocks Swap setting. Users who saved an explicit value keep it.
    first_run = not os.path.exists(PREFS_FILE)
    _ = user_set_swap  # retained for clarity; no longer forces a concrete value
    # Resolve portable directory paths to absolute.
    for key in _PORTABLE_DIR_KEYS:
        if key in prefs and isinstance(prefs[key], str):
            prefs[key] = _resolve_pref_path(prefs[key])
    # On first run, persist defaults so prefs.json exists immediately.
    if first_run:
        save_prefs(prefs)
    return prefs


def _persist_disabled() -> bool:
    """True when FIZGIG_NO_PERSIST is set — headless test harnesses set it so instantiating the
    GUI and poking vars can NEVER overwrite the user's real prefs.json / last_used.json /
    settings / Fizgig_train.toml (traced vars auto-save on write, so a test setting
    image_folder_var would otherwise clobber the remembered training folder)."""
    return bool(os.environ.get("FIZGIG_NO_PERSIST"))


def _running_on_pod() -> bool:
    """True when this is the Docker pod image (RunPod, or any rented box).

    Keyed off OUR marker, set in docker/entrypoint.sh, rather than RunPod's RUNPOD_POD_ID: the
    hosting provider's variable names are theirs to change, and another host would set different
    ones entirely. The pod id below is read separately and only used for display and for targeting
    a stop — everything degrades if it is absent."""
    return os.environ.get("FIZGIG_POD", "0") not in ("0", "", None)


def _pod_id() -> str:
    """The provider's id for this pod, or "" if it doesn't advertise one."""
    for key in ("RUNPOD_POD_ID", "HOSTNAME"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    return ""


def _app_commit() -> str:
    """Short git commit of the running checkout, or "".

    On a pod this is the interesting half of the version: the IMAGE is pinned in the RunPod
    template while the app pulls master at every boot, so the two are meant to differ and a bug
    report needs both."""
    try:
        import subprocess as _sp
        r = _sp.run(["git", "-C", _FIZGIG_DIR, "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=8,
                    creationflags=(0x08000000 if os.name == "nt" else 0))
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _git(*args, timeout=8) -> str:
    """Run a git command in the repo, return stripped stdout or "" on any failure."""
    try:
        import subprocess as _sp
        r = _sp.run(["git", "-C", _FIZGIG_DIR, *args],
                    capture_output=True, text=True, timeout=timeout,
                    creationflags=(0x08000000 if os.name == "nt" else 0))
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _git_describe_version() -> str:
    """Human version of the running checkout: 'v3.1.1' exactly on a tag,
    'v3.1.1-2-gee3a7fa' in between. Pods clone --depth 1 (no tags), so describe falls back
    to a bare short SHA — which read as a mystery build number in the field ('Version
    37c0c2f' was current master, taken for an old app). Name the branch so a tagless
    checkout says what it is: 'master @ 37c0c2f'."""
    v = _git("describe", "--tags", "--always") or _app_commit()
    if v and "v" not in v.split("-")[0]:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        if branch and branch != "HEAD":
            return f"{branch} @ {v}"
    return v


def _latest_release_tag():
    """Highest vX.Y.Z release tag on origin as (tag, sha), or None when unreachable.

    ls-remote is a single read-only round-trip — no fetch, no GitHub API, no tokens or
    rate limits. Returning None (offline, no origin, ZIP download) means 'unknown':
    the caller shows nothing rather than a false nag."""
    out = _git("ls-remote", "--tags", "--refs", "origin", "v*", timeout=15)
    if not out:
        return None
    best, best_key = None, None
    for line in out.splitlines():
        try:
            sha, ref = line.split(None, 1)
            tag = ref.strip().rsplit("/", 1)[-1]
            key = tuple(int(p) for p in tag.lstrip("v").split("."))
        except (ValueError, IndexError):
            continue    # not a plain vX.Y.Z tag — ignore
        if best_key is None or key > best_key:
            best, best_key = (tag, sha), key
    return best


def _update_status_from(latest, has_obj: bool, is_ancestor: bool) -> str:
    """Pure decision: 'up_to_date' | 'update_available' | 'unknown'.

    Up to date means HEAD CONTAINS the latest release tag's commit — true both exactly on
    the tag and ahead of it (a user who pulled newer master must not be nagged). A tag whose
    commit we don't even have locally, or have but isn't an ancestor, means the release is
    newer than this checkout."""
    if latest is None:
        return "unknown"
    if has_obj and is_ancestor:
        return "up_to_date"
    return "update_available"


def _git_ok(*args, timeout=8) -> bool:
    """Run git for its EXIT CODE (merge-base --is-ancestor answers that way)."""
    try:
        import subprocess as _sp
        r = _sp.run(["git", "-C", _FIZGIG_DIR, *args],
                    capture_output=True, text=True, timeout=timeout,
                    creationflags=(0x08000000 if os.name == "nt" else 0))
        return r.returncode == 0
    except Exception:
        return False


def _check_for_update():
    """Full check (network + local git). Returns ('update_available', tag) /
    ('up_to_date', current) / ('unknown', '')."""
    latest = _latest_release_tag()
    if latest is None:
        return "unknown", ""
    tag, sha = latest
    has_obj = _git_ok("cat-file", "-e", f"{sha}^{{commit}}")
    is_anc = has_obj and _git_ok("merge-base", "--is-ancestor", sha, "HEAD")
    status = _update_status_from(latest, has_obj, is_anc)
    if status == "update_available":
        # Shallow-clone truth (pods clone --depth 1): the tag's commit isn't in the local
        # object store even when HEAD is AHEAD of it, so ancestry can't clear us — but being
        # exactly the remote tip can. Without this, every pod born after any post-release
        # commit showed a FALSE Update Available banner on perfectly current code (the 19 Aug
        # "old pods" saga — three images, all current, all nagging). A genuinely stale pod
        # is neither on the tag nor at the tip, so real updates still flag.
        remote_head = (_git("ls-remote", "origin", "HEAD", timeout=15) or "").split()
        local_head = _git("rev-parse", "HEAD")
        if remote_head and local_head and remote_head[0] == local_head:
            status = "up_to_date"
    return status, (tag if status == "update_available" else _git_describe_version())


def _pod_stop_key_env() -> str:
    """A stop-capable key from the environment, or "".

    NOT RunPod's injected RUNPOD_API_KEY — that one is pod-scoped and 403s on every pod-management
    call (verified on a live pod, and a documented RunPod limitation). Stopping a pod needs an
    account key, which is why this reads a separate variable rather than falling back."""
    return (os.environ.get("RUNPOD_STOP_API_KEY") or "").strip()


def save_prefs(prefs: dict) -> None:
    """Save preferences to prefs.json. Portable-dir paths inside the repo are
    stored as relative strings so a cloned/moved repo finds its own defaults."""
    if _persist_disabled():
        return
    to_save = {}
    for key, value in prefs.items():
        if key in _PORTABLE_DIR_KEYS and isinstance(value, str):
            to_save[key] = _serialize_pref_path(value)
        else:
            to_save[key] = value
    try:
        # Atomic: a truncated prefs.json silently blanked every model path (the reader
        # swallows JSONDecodeError and falls back to defaults, then the next auto-save
        # persisted the blanks).
        tmp = PREFS_FILE + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2)
        os.replace(tmp, PREFS_FILE)
    except Exception:
        pass