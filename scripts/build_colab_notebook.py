"""Programmatic builder for colab/cavr_e2e.ipynb.

Run locally:

    python scripts/build_colab_notebook.py
    jupyter nbconvert --to script colab/cavr_e2e.ipynb --stdout | head -300

We never hand-write .ipynb JSON. The build script is the source of truth;
the notebook is the artifact (committed for Colab convenience). Re-run this
script after every notebook change.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat as nbf


REPO_URL = "https://github.com/javaapplesauce/r3m-4-DLRM.git"
OUT_PATH = Path("colab/cavr_e2e.ipynb")


def _md(source: str, tag: str):
    cell = nbf.v4.new_markdown_cell(source)
    cell.metadata["tags"] = [tag]
    return cell


def _code(source: str, tag: str):
    cell = nbf.v4.new_code_cell(source)
    cell.metadata["tags"] = [tag]
    return cell


def build_cells() -> list:
    cells: list = []

    cells.append(_md(
        "# CAVR end-to-end (Colab Pro, A100)\n"
        "\n"
        "Concept-Aware Visual Representations for Robotic Manipulation.\n"
        "Frozen DINOv2 ViT-L/14 dense features, filtered by a Grounding DINO →\n"
        "SAM 2 concept mask, spatial-mean-pooled, concatenated with 14-d\n"
        "proprioception, and fed through a 3-layer MLP for 6-DOF action.\n"
        "Baselines: R3M (ResNet-50) and VC-1 (ViT-L). Tasks: `Lift`, `PickPlace`.\n"
        "Seeds: 0, 1, 2. 18 baseline runs + 6 ablation runs.\n"
        "\n"
        "## HOW TO RUN\n"
        "\n"
        "1. **Set the runtime** to A100: *Runtime → Change runtime type → A100*.\n"
        "2. Run cells **top to bottom**. Cells 1–7 are setup (idempotent).\n"
        "3. If the runtime disconnects:\n"
        "   - Re-run cells 1–7 (`git pull` keeps the repo current).\n"
        "   - Run the *resume status* cell to see what's left.\n"
        "   - Continue from the training sweep cell — runs are cached on Drive\n"
        "     per `(model, task, seed)`, so completed work is skipped.\n"
        "4. Mid-sweep bug? I fix locally, push, you re-run cell 1 (pulls fix)\n"
        "   then jump straight back to the sweep cell."
    , "title"))

    cells.append(_code(
        f"""# Cell 1: clone or pull the repo. Idempotent.
REPO_URL = "{REPO_URL}"
WORK_DIR = "/content/cavr"

import os, subprocess
if os.path.exists(WORK_DIR):
    subprocess.run(["git", "-C", WORK_DIR, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True)
os.chdir(WORK_DIR)
HEAD = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
print("HEAD:", HEAD)
"""
    , "setup-repo"))

    cells.append(_code(
        """# Cell 2: GPU sanity check.
import torch
assert torch.cuda.is_available(), "No GPU! Set runtime to A100 (Runtime → Change runtime type)."
name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", name)
print(f"VRAM: {vram_gb:.1f} GB")
print("CUDA:", torch.version.cuda)
print("PyTorch:", torch.__version__)

OK_GPUS = ("A100", "L4", "H100", "A100-40GB", "A100-80GB")
if not any(tag in name for tag in OK_GPUS):
    print("\\n" + "!" * 60)
    print(f"WARNING: GPU '{name}' is slower than the sweep was sized for.")
    print("The sweep will still run, but expect 2-5x the wallclock.")
    print("!" * 60 + "\\n")
"""
    , "setup-gpu"))

    cells.append(_code(
        """# Cell 3: mount Drive. Everything persistent lives under DRIVE_ROOT.
from google.colab import drive
import os
drive.mount("/content/drive")

DRIVE_ROOT = "/content/drive/MyDrive/cavr_runs"
os.makedirs(DRIVE_ROOT, exist_ok=True)
print("Drive root:", DRIVE_ROOT)

for root, dirs, files in os.walk(DRIVE_ROOT):
    depth = root[len(DRIVE_ROOT):].count(os.sep)
    if depth < 2:
        rel = os.path.relpath(root, DRIVE_ROOT) if root != DRIVE_ROOT else "."
        print(" " * depth + (rel if rel != "." else "(root)") + "/")
        for fn in files[:6]:
            print(" " * (depth + 2) + fn)
"""
    , "setup-drive"))

    cells.append(_code(
        """# Cell 4: symlink data / outputs / checkpoints into Drive.
# Anything we care about persisting goes through one of these dirs.
import os, shutil

WORK_DIR = "/content/cavr"
DRIVE_ROOT = "/content/drive/MyDrive/cavr_runs"

for name in ("data", "outputs", "checkpoints"):
    drive_target = os.path.join(DRIVE_ROOT, name)
    local_link = os.path.join(WORK_DIR, name)
    os.makedirs(drive_target, exist_ok=True)

    if os.path.islink(local_link):
        # Already symlinked from a prior session — leave it.
        continue
    if os.path.exists(local_link):
        # A real directory from a fresh clone. Move its contents (if any)
        # to Drive, then replace the directory with a symlink.
        for entry in os.listdir(local_link):
            src = os.path.join(local_link, entry)
            dst = os.path.join(drive_target, entry)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(local_link, ignore_errors=True)
    os.symlink(drive_target, local_link)

print("Symlinks:")
for name in ("data", "outputs", "checkpoints"):
    p = os.path.join(WORK_DIR, name)
    real = os.path.realpath(p)
    print(f"  {p}  ->  {real}")
"""
    , "setup-symlinks"))

    cells.append(_code(
        """# Cell 5: OS-level deps for MuJoCo + EGL.
# Quiet output: the apt-get firehose isn't useful unless something fails.
!apt-get install -y \\
    libosmesa6-dev libgl1-mesa-glx libglfw3 libegl1 \\
    libegl1-mesa-dev libgles2-mesa-dev xvfb 2>&1 | tail -5
"""
    , "setup-apt"))

    cells.append(_code(
        """# Cell 6: Python deps. Split into chunks so a single failure is easy to spot.
# We deliberately do NOT use -q here: sam2 and r3m build CUDA/native bits and
# their actual error messages are the only useful debug signal when they fail.
import subprocess, sys

def pip(*args, allow_fail=False):
    cmd = [sys.executable, "-m", "pip", "install"] + list(args)
    print("$", " ".join(cmd[3:]))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        if allow_fail:
            print(f"  [warn] {' '.join(args)} failed (rc={e.returncode}); continuing.")
        else:
            raise

# 1) The package itself (CAVR), with the [all] extra. Torch stays as Colab ships it.
pip("-e", ".[all]")

# 2) Concept masking stack. transformers is in [all]; sam2 is intentionally not.
# Use the plain git URL (not "name @ url") — the PEP 508 name-prefix form can
# trip on package-name-vs-distribution-name mismatches in some pip versions.
pip("git+https://github.com/facebookresearch/sam2.git")

# 3) Baselines. VC-1 is optional — baseline falls back to a timm ViT-L.
# CRITICAL: install r3m with --no-deps. r3m's setup.py pins gym==0.21.0
# which silently downgrades numpy to 1.x, breaking the h5py / mujoco /
# scipy wheels Colab ships compiled against numpy 2.x. At runtime r3m
# only needs torch + torchvision (already present), so --no-deps is safe.
pip("--no-deps", "git+https://github.com/facebookresearch/r3m.git")
# r3m's load_r3m calls into a couple of small utility packages that we
# do need; install them by hand without letting them re-pull gym.
pip("hydra-core", "omegaconf", "huggingface_hub", allow_fail=True)
pip("git+https://github.com/facebookresearch/eai-vc.git#subdirectory=vc_models",
    allow_fail=True)

# Defensive numpy pin: Colab ships scipy/h5py/mujoco/robosuite wheels built
# against numpy 2.0.x. If anything earlier in the install chain bumped numpy
# even one minor version, scipy import dies with "cannot import _center from
# numpy._core.umath". Force numpy back to Colab's default. (Requires a kernel
# restart on first install to take effect — Colab does that implicitly when
# numpy is reinstalled, but if you see ImportErrors after this cell, manually
# Runtime → Restart session and re-run from Cell 1.)
pip("--force-reinstall", "--no-deps", "numpy==2.0.2")

# 3b) Upgrade timm. Grounding DINO transitively imports timm, and
# Colab's stock timm has a dataclass mutable-default bug
# (timm.models.maxxvit.MaxxVitConvCfg) that Python 3.11+ rejects at
# import time. Without the upgrade, the mask precompute silently falls
# back to all-ones masks and CAVR collapses to a no-mask baseline.
pip("--upgrade", "timm>=1.0.7")

# 4) SAM 2 checkpoint. -nc skipped if file already exists; otherwise wget
# without -q so a failed download is visible. Verify size > 100 MB.
import os
os.makedirs("checkpoints", exist_ok=True)
need_download = True
if os.path.exists("sam2_hiera_large.pt"):
    sz = os.path.getsize("sam2_hiera_large.pt")
    if sz > 100_000_000:
        print(f"  sam2 ckpt OK ({sz/1e6:.0f} MB)")
        need_download = False
    else:
        print(f"  sam2 ckpt too small ({sz} bytes); re-downloading.")
        os.remove("sam2_hiera_large.pt")
if need_download:
    !wget --show-progress -O sam2_hiera_large.pt \\
        https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    sz = os.path.getsize("sam2_hiera_large.pt")
    assert sz > 100_000_000, f"sam2 ckpt download failed; only got {sz} bytes."

# 5) Import smoke test. Each line is one import; failures stay visible.
import importlib, traceback
modules = ["cavr", "robosuite", "transformers", "sam2", "r3m"]
optional = ["vc_models"]
for m in modules + optional:
    try:
        importlib.import_module(m)
        print(f"  [ok] {m}")
    except Exception as e:
        flag = "WARN" if m in optional else "FAIL"
        print(f"  [{flag}] {m}: {e}")

# DINOv2 isn't a pip package; it's loaded via torch.hub. Smoke-test that path.
try:
    import torch
    _ = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
    print("  [ok] dinov2 (torch.hub)")
except Exception as e:
    print(f"  [FAIL] dinov2 hub load: {e}")
"""
    , "setup-pip"))

    cells.append(_code(
        """# Cell 7: EGL backend + robosuite smoke test (offscreen render).
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import robosuite
env = robosuite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=224,
    camera_widths=224,
)
obs = env.reset()
print("robosuite OK, obs keys (first 8):", list(obs.keys())[:8])
env.close()
"""
    , "setup-egl"))

    cells.append(_md(
        "## Section B: Data preparation\n"
        "\n"
        "Idempotent. Skips collection when a populated `demos.hdf5` is on Drive\n"
        "(≥30 demos), skips mask precompute when `masks.hdf5` is present.\n"
        "\n"
        "**Time-crunched?** Set `QUICK_MODE = True` in the next cell to limit\n"
        "the sweep to Lift only (skips PickPlace data, halves the runtime budget)."
    , "section-data"))

    cells.append(_code(
        """# Cell 8b: sweep configuration. Edit in one place; downstream cells read these.
QUICK_MODE = False         # True → Lift only, fewer epochs / seeds / eval episodes.

if QUICK_MODE:
    TASKS = ["Lift"]
    SEEDS = [0, 1, 2]
    EPOCHS = 40
    EVAL_EPISODES = 20
    BATCH_SIZE = 64
    EVAL_FREQ = 5
    RUN_ABLATION = True
else:
    TASKS = ["Lift", "PickPlace"]
    SEEDS = [0, 1, 2]
    EPOCHS = 60
    EVAL_EPISODES = 25
    BATCH_SIZE = 64
    EVAL_FREQ = 5
    RUN_ABLATION = True

print(f"TASKS={TASKS}  SEEDS={SEEDS}  EPOCHS={EPOCHS}  "
      f"EVAL_EPISODES={EVAL_EPISODES}  RUN_ABLATION={RUN_ABLATION}")
"""
    , "sweep-config"))

    cells.append(_code(
        """# Cell 9: collect scripted demos for each task.
# In-process call so errors land in the cell instead of being swallowed by
# a subprocess. Idempotent: skips tasks with ≥30 demos already cached.
import copy
import os
import traceback

import h5py
import yaml

from cavr.data.collector import collect_scripted_demos

TARGET_DEMOS = 50
MIN_DEMOS = 30

with open("cavr/configs/default.yaml") as f:
    BASE_CFG = yaml.safe_load(f)

for task in TASKS:
    task_dir = f"data/demos_{task}"
    demo_file = f"{task_dir}/demos.hdf5"

    n = 0
    if os.path.exists(demo_file):
        try:
            with h5py.File(demo_file, "r") as f:
                n = len(list(f.keys()))
        except Exception:
            n = 0
    if n >= MIN_DEMOS:
        print(f"[skip] {task}: {n} demos cached at {demo_file}")
        continue

    print(f"[collect] {task}: target {TARGET_DEMOS} demos (have {n}) → {task_dir}")
    cfg = copy.deepcopy(BASE_CFG)
    cfg["env"]["name"] = task
    cfg["data"]["num_demos"] = TARGET_DEMOS
    cfg["data"]["save_dir"] = task_dir
    try:
        collect_scripted_demos(cfg)
    except Exception as e:
        print(f"[FAIL] {task}: {e}")
        traceback.print_exc()
        continue

    with h5py.File(demo_file, "r") as f:
        actual = len(list(f.keys()))
    print(f"[collect] {task}: wrote {actual} demos to {demo_file}")
    if actual < MIN_DEMOS:
        print(f"!!! {task} only produced {actual} (<{MIN_DEMOS}). Stop and ping "
              f"Claude Code — the scripted collector heuristic is task-dependent.")
"""
    , "data-demos"))

    cells.append(_code(
        """# Cell 10: precompute Grounding-DINO + SAM2 masks per task.
# In-process call — errors hit the cell directly.
import os
import traceback

from scripts.precompute_masks import precompute
from cavr.utils.reporting import mask_coverage_stats

for task in TASKS:
    task_dir = f"data/demos_{task}"
    mask_file = f"{task_dir}/masks.hdf5"

    if os.path.exists(mask_file):
        print(f"[skip] {task}: masks cached at {mask_file}")
    else:
        if not os.path.exists(f"{task_dir}/demos.hdf5"):
            print(f"[skip] {task}: no demos at {task_dir}/demos.hdf5")
            continue
        print(f"[masks] {task}: precomputing → {mask_file}")
        try:
            precompute(data_dir=task_dir, env_name=task)
        except Exception as e:
            print(f"[FAIL] {task}: {e}")
            traceback.print_exc()
            continue

    if os.path.exists(mask_file):
        frac, n = mask_coverage_stats(mask_file)
        print(f"  {task}: mean coverage = {frac:.3f}, frames = {n}")
        if frac < 0.01 or frac > 0.5:
            print("  !!! WARNING: coverage outside [0.01, 0.5]. "
                  "Likely empty / failed / over-broad masks.")
"""
    , "data-masks"))

    cells.append(_code(
        """# Cell 11: spot-check that masks actually look right.
import os
os.makedirs("outputs/figures", exist_ok=True)

import matplotlib.pyplot as plt
from cavr.utils.reporting import render_spotcheck_panel

for task in TASKS:
    demos = f"data/demos_{task}/demos.hdf5"
    masks = f"data/demos_{task}/masks.hdf5"
    if not os.path.exists(demos):
        print(f"[skip] {task}: no demos at {demos}")
        continue
    out_png = f"outputs/figures/spotcheck_{task}.png"
    fig = render_spotcheck_panel(demos, masks if os.path.exists(masks) else None,
                                 task=task, n=8, out_png=out_png, seed=0)
    plt.show()
    plt.close(fig)
    print(f"saved {out_png}")

print("\\n" + "=" * 60)
print("REVIEW THE MASKS ABOVE. If they look wrong (empty, on the wrong")
print("object, or covering everything), STOP HERE before training.")
print("=" * 60)
"""
    , "data-spotcheck"))

    cells.append(_md(
        "## Resume status\n"
        "\n"
        "Run this any time to see which `(model, task, seed)` runs are still\n"
        "outstanding. Reads `outputs/runs/*.json` on Drive, so it works after\n"
        "a disconnect."
    , "section-resume"))

    cells.append(_code(
        """# Cell 13: print remaining runs. Uses TASKS/SEEDS from the sweep-config cell.
from itertools import product
from cavr.utils.reporting import print_resume_status

BASELINE_MODELS = ["cavr", "r3m", "vc1"]

baseline_expected = list(product(BASELINE_MODELS, TASKS, SEEDS))
print(f"BASELINES ({len(baseline_expected)} runs)")
print_resume_status("outputs/runs", baseline_expected)

ABLATION_VARIANTS = [
    "cavr_vitl_masked",
    "cavr_vitl_no_mask",
    # ViT-B variants are only run if there's wallclock left.
    "cavr_vitb_masked",
    "cavr_vitb_no_mask",
]
ablation_expected = [(v, s) for v in ABLATION_VARIANTS for s in SEEDS]
print(f"\\nABLATIONS ({len(ablation_expected)} runs scheduled; ViT-B optional)")
print_resume_status("outputs/runs", ablation_expected)
"""
    , "resume-status"))

    cells.append(_md(
        "## Section C: Training sweep\n"
        "\n"
        "18 baseline runs (3 models × 2 tasks × 3 seeds) + 6 ablation runs\n"
        "(CAVR ViT-L masked vs no-mask on Lift, 3 seeds; ViT-B variants if\n"
        "time remains). Each completed run lands as one JSON file on Drive,\n"
        "and one CSV row. Skip-if-cached at `(model, task, seed)` granularity.\n"
        "\n"
        "**Wallclock budget:** ~5–7 h on A100 for the full sweep. If the\n"
        "runtime disconnects, re-run setup cells 1–7 then this cell — it picks\n"
        "up where it left off."
    , "section-training"))

    cells.append(_code(
        """# Cell 15: baseline sweep. {cavr, r3m, vc1} × TASKS × SEEDS.
# Uses TASKS / SEEDS / EPOCHS / EVAL_EPISODES from the sweep-config cell.
# train_and_eval is skip-if-cached at (model, task, seed) granularity, so
# re-running this cell after a disconnect picks up where it stopped.
import time
from itertools import product

from cavr.utils.runs import train_and_eval

NOTEBOOK_START_TIME = time.time()
BASELINE_MODELS = ["cavr", "r3m", "vc1"]

for model, task, seed in product(BASELINE_MODELS, TASKS, SEEDS):
    train_and_eval(
        model, task, seed,
        runs_dir="outputs/runs",
        csv_path="outputs/baseline_results.csv",
        epochs=EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
        batch_size=BATCH_SIZE,
    )

print(f"\\nBaseline sweep elapsed: {(time.time() - NOTEBOOK_START_TIME)/60:.1f} min")
"""
    , "training-baselines"))

    cells.append(_code(
        """# Cell 16: ablation sweep on Lift (always Lift, regardless of QUICK_MODE).
# Required: cavr_vitl_{masked, no_mask}. Optional: cavr_vitb_{masked, no_mask}.
import time
from itertools import product

from cavr.utils.runs import train_and_eval

if not RUN_ABLATION:
    print("[skip] RUN_ABLATION=False (set in the sweep-config cell)")
else:
    # Resilient to cell 15 not having run this session (post-disconnect path).
    try:
        NOTEBOOK_START_TIME
    except NameError:
        NOTEBOOK_START_TIME = time.time()
    WALLCLOCK_CAP_HOURS = 9.0

    VITL_VARIANTS = [
        ("cavr_vitl_masked",  "dinov2_vitl14", True),
        ("cavr_vitl_no_mask", "dinov2_vitl14", False),
    ]
    VITB_VARIANTS = [
        ("cavr_vitb_masked",  "dinov2_vitb14", True),
        ("cavr_vitb_no_mask", "dinov2_vitb14", False),
    ]

    def run_variant(variant, backbone, masked, seed):
        train_and_eval(
            "cavr", "Lift", seed,
            runs_dir="outputs/runs",
            csv_path="outputs/ablation_results.csv",
            epochs=EPOCHS,
            eval_freq=EVAL_FREQ,
            eval_episodes=EVAL_EPISODES,
            batch_size=BATCH_SIZE,
            masking_override=masked,
            backbone_override=backbone,
            variant=f"{variant}_seed{seed}",
        )

    for (name, backbone, masked), seed in product(VITL_VARIANTS, SEEDS):
        run_variant(name, backbone, masked, seed)

    elapsed_h = (time.time() - NOTEBOOK_START_TIME) / 3600
    if elapsed_h > WALLCLOCK_CAP_HOURS:
        print(f"\\n[budget] elapsed={elapsed_h:.2f}h > {WALLCLOCK_CAP_HOURS}h — "
              f"skipping ViT-B variants.")
    else:
        for (name, backbone, masked), seed in product(VITB_VARIANTS, SEEDS):
            run_variant(name, backbone, masked, seed)

    print(f"\\nAblation sweep elapsed: {(time.time() - NOTEBOOK_START_TIME)/60:.1f} min")
"""
    , "training-ablation"))

    cells.append(_md(
        "## Section D: Figures\n"
        "\n"
        "Built from the cached CSVs and JSONs. Each figure cell is re-runnable\n"
        "after tweaks. PDFs go to `outputs/figures/` (mirrored to Drive)."
    , "section-figures"))

    cells.append(_code(
        """# Cell 18: Fig 1 — headline bar chart (CAVR vs R3M vs VC-1, per task).
# Tolerates partial data: missing (model, task) cells render as zero-height
# bars with no error, and tasks with no rows are dropped entirely.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif"})

CSV_PATH = "outputs/baseline_results.csv"
if not os.path.exists(CSV_PATH):
    print(f"[skip] {CSV_PATH} doesn't exist yet — no figure to build.")
    raise SystemExit
df = pd.read_csv(CSV_PATH)
df = df[df["status"] == "DONE"].copy()
df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
df = df.dropna(subset=["success_rate"])
if df.empty:
    print("[skip] no completed baseline runs in the CSV yet.")
    raise SystemExit

MODELS = ["cavr", "r3m", "vc1"]
# Only plot tasks we actually have data for.
TASKS_HERE = [t for t in ["Lift", "PickPlace"] if (df["task"] == t).any()]
COLORS = {"cavr": "#1f77b4", "r3m": "#ff7f0e", "vc1": "#2ca02c"}

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(TASKS_HERE))
width = 0.25

for i, model in enumerate(MODELS):
    means, stds = [], []
    for task in TASKS_HERE:
        sub = df[(df["model"] == model) & (df["task"] == task)]
        if len(sub) == 0:
            means.append(0.0); stds.append(0.0)
        else:
            means.append(sub["success_rate"].mean())
            stds.append(sub["success_rate"].std(ddof=0))
    bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4,
                  label=model.upper(), color=COLORS[model])
    for j, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{means[j]:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(TASKS_HERE)
ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.0)
ax.set_title("CAVR vs visual-representation baselines (3 seeds)")
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/figures/fig1_headline.pdf")
plt.show()
"""
    , "figures-1-headline"))

    cells.append(_code(
        """# Cell 19: Fig 2 — ablation (masked vs no-mask, ViT-L and optionally ViT-B).
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif"})

CSV_PATH = "outputs/ablation_results.csv"
if not os.path.exists(CSV_PATH):
    print(f"[skip] {CSV_PATH} doesn't exist — ablation cell never ran.")
    raise SystemExit
df = pd.read_csv(CSV_PATH)
df = df[df["status"] == "DONE"].copy()
df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
df = df.dropna(subset=["success_rate"])
if df.empty:
    print("[skip] no completed ablation runs yet.")
    raise SystemExit

variants = df["variant"].dropna().unique().tolist() if "variant" in df.columns else []

def stats(name):
    sub = df[df["variant"] == name] if "variant" in df.columns else df[df["run_id"].str.startswith(name)]
    if len(sub) == 0:
        return None
    return sub["success_rate"].mean(), sub["success_rate"].std(ddof=0)

vitl_m = stats("cavr_vitl_masked")
vitl_n = stats("cavr_vitl_no_mask")
vitb_m = stats("cavr_vitb_masked")
vitb_n = stats("cavr_vitb_no_mask")

has_vitb = vitb_m is not None and vitb_n is not None

fig, ax = plt.subplots(figsize=(7, 4.5))

if has_vitb:
    labels = ["ViT-L", "ViT-B"]
    masked = [vitl_m[0], vitb_m[0]]
    masked_err = [vitl_m[1], vitb_m[1]]
    nomask = [vitl_n[0], vitb_n[0]]
    nomask_err = [vitl_n[1], vitb_n[1]]
    x = np.arange(len(labels)); w = 0.35
    b1 = ax.bar(x - w / 2, masked, w, yerr=masked_err, capsize=4, label="masked", color="#1f77b4")
    b2 = ax.bar(x + w / 2, nomask, w, yerr=nomask_err, capsize=4, label="no mask", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for bars, vals in [(b1, masked), (b2, nomask)]:
        for j, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{vals[j]:.2f}", ha="center", va="bottom", fontsize=9)
else:
    labels = ["masked", "no mask"]
    means = [vitl_m[0] if vitl_m else 0.0, vitl_n[0] if vitl_n else 0.0]
    errs = [vitl_m[1] if vitl_m else 0.0, vitl_n[1] if vitl_n else 0.0]
    bars = ax.bar(labels, means, yerr=errs, capsize=4,
                  color=["#1f77b4", "#d62728"])
    for j, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{means[j]:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.0)
ax.set_title("Ablation: concept masking on CAVR (Lift, 3 seeds)")
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/figures/fig2_ablation.pdf")
plt.show()
"""
    , "figures-2-ablation"))

    cells.append(_code(
        """# Cell 20: Fig 3 — training curves (per task, mean ± std across seeds).
import os
import numpy as np
import matplotlib.pyplot as plt

from cavr.utils.io import load_run_jsons

plt.rcParams.update({"font.family": "serif"})

df = load_run_jsons("outputs/runs/*.json")
df = df[df.get("status") == "DONE"]

MODELS = ["cavr", "r3m", "vc1"]
TASKS = ["Lift", "PickPlace"]
COLORS = {"cavr": "#1f77b4", "r3m": "#ff7f0e", "vc1": "#2ca02c"}

fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.5), sharey=True)
for ax, task in zip(axes, TASKS):
    for model in MODELS:
        sub = df[(df["model"] == model) & (df["task"] == task)]
        curves = []
        for _, row in sub.iterrows():
            vls = row.get("val_losses") or []
            if not vls:
                continue
            epochs, losses = zip(*vls)
            curves.append((np.asarray(epochs), np.asarray(losses, dtype=float)))
        if not curves:
            continue
        epochs_ref = curves[0][0]
        stacked = np.stack([c[1] for c in curves if len(c[1]) == len(epochs_ref)])
        if stacked.size == 0:
            continue
        mean = stacked.mean(0); std = stacked.std(0)
        ax.plot(epochs_ref, mean, label=model.upper(), color=COLORS[model])
        ax.fill_between(epochs_ref, mean - std, mean + std, alpha=0.2, color=COLORS[model])
    ax.set_title(task); ax.set_xlabel("epoch")
    if ax is axes[0]:
        ax.set_ylabel("val loss")
    ax.spines[["top", "right"]].set_visible(False)

# Log y if range spans >10x.
all_vals = []
for _, row in df.iterrows():
    vls = row.get("val_losses") or []
    if vls:
        all_vals.extend(v for _, v in vls)
if all_vals and max(all_vals) > 10 * (min(all_vals) + 1e-9):
    for ax in axes:
        ax.set_yscale("log")

axes[0].legend(loc="upper right")
plt.tight_layout()
plt.savefig("outputs/figures/fig3_training_curves.pdf")
plt.show()
"""
    , "figures-3-curves"))

    cells.append(_code(
        """# Cell 21: Fig 4 — qualitative mask panel. 3 frames per task.
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from cavr.utils.reporting import _resize_nearest

plt.rcParams.update({"font.family": "serif"})

TASKS = ["Lift", "PickPlace"]
N_PER_TASK = 3
rng = np.random.default_rng(7)

fig, axes = plt.subplots(2 * N_PER_TASK, 4, figsize=(12, 3 * 2 * N_PER_TASK))

row = 0
for task in TASKS:
    demos_p = f"data/demos_{task}/demos.hdf5"
    masks_p = f"data/demos_{task}/masks.hdf5"
    if not os.path.exists(demos_p):
        continue

    with h5py.File(demos_p, "r") as df:
        keys = sorted(df.keys())
        pool = [(k, t) for k in keys for t in range(df[k]["images"].shape[0])]
        picks = [pool[i] for i in rng.choice(len(pool), size=N_PER_TASK, replace=False)]
        rgbs = []
        for k, t in picks:
            img = df[k]["images"][t]
            if img.shape[0] == 3 and img.ndim == 3:
                img = img.transpose(1, 2, 0)
            rgbs.append(img.astype(np.uint8))

    masks_full = [None] * N_PER_TASK
    if os.path.exists(masks_p):
        with h5py.File(masks_p, "r") as mf:
            for i, (k, t) in enumerate(picks):
                if k in mf:
                    masks_full[i] = np.asarray(mf[k]["masks"][t], dtype=np.uint8)

    for i, (rgb, mask) in enumerate(zip(rgbs, masks_full)):
        H, W = rgb.shape[:2]
        ax = axes[row, 0]; ax.imshow(rgb); ax.axis("off")
        if i == 0:
            ax.set_title(f"{task}: RGB", loc="left")
        if mask is not None and mask.any():
            mask_up = _resize_nearest(mask, (H, W))
            ys, xs = np.where(mask_up > 0)
            x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
            axes[row, 1].imshow(rgb)
            axes[row, 1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             linewidth=2, edgecolor="lime", facecolor="none"))
            axes[row, 1].axis("off")
            axes[row, 2].imshow(mask_up, cmap="gray", vmin=0, vmax=1); axes[row, 2].axis("off")
            overlay = rgb.astype(np.float32).copy()
            sel = mask_up > 0
            overlay[sel] = 0.5 * overlay[sel] + 0.5 * np.array([0.0, 255.0, 0.0])
            axes[row, 3].imshow(overlay.clip(0, 255).astype(np.uint8)); axes[row, 3].axis("off")
        else:
            for c in range(1, 4):
                axes[row, c].imshow(rgb); axes[row, c].axis("off")
        row += 1

fig.suptitle("Qualitative concept-mask outputs", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/figures/fig4_qualitative_masks.pdf")
plt.show()
"""
    , "figures-4-masks"))

    cells.append(_code(
        """# Cell 22: bundle figures + CSVs + manifest into one zip, copy to Drive.
import json
import os
import shutil
import time
import zipfile
from glob import glob

from cavr.utils.io import git_short_hash, gpu_name, load_run_jsons

manifest = {
    "git_hash": git_short_hash(),
    "gpu": gpu_name(),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "runs": {},
}

df = load_run_jsons("outputs/runs/*.json")
attempted = len(df)
succeeded = int((df.get("status") == "DONE").sum()) if attempted else 0
failed = attempted - succeeded
manifest["runs"] = {
    "attempted": attempted,
    "succeeded": succeeded,
    "failed": failed,
}

manifest_path = "outputs/manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)

bundle_path = "outputs/cavr_results_bundle.zip"
with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("outputs/figures/*.pdf", "outputs/figures/*.png",
                "outputs/*.csv", "outputs/manifest.json"):
        for p in sorted(glob(pat)):
            z.write(p, arcname=os.path.relpath(p, "outputs"))

drive_bundle = "/content/drive/MyDrive/cavr_runs/cavr_results_bundle.zip"
shutil.copy(bundle_path, drive_bundle)

print("=" * 60)
print(f"Runs attempted: {attempted}  succeeded: {succeeded}  failed: {failed}")
print(f"Bundle:         {drive_bundle}")
print(f"Manifest:       {manifest_path}")
print("=" * 60)
print(json.dumps(manifest, indent=2, default=str))
"""
    , "figures-bundle"))

    cells.append(_md(
        "## If you disconnected and need to recover\n"
        "\n"
        "1. Re-run cells 1–7 (setup). Idempotent — `git pull` brings the repo\n"
        "   current and `pip install -e .` rebinds the package.\n"
        "2. Skip cells 9–11 (data prep). Demos + masks are already on Drive.\n"
        "3. Run cell 13 (resume status) to see what's left.\n"
        "4. Run cells 15–16 (training). The orchestrator skips runs whose JSON\n"
        "   is already complete on Drive.\n"
        "5. Once the sweep is done, run cells 18–22 to regenerate figures and\n"
        "   the bundle."
    , "section-recovery"))

    return cells


def main():
    cells = build_cells()
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "A100"}

    nbf.validate(nb)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        nbf.write(nb, f)

    print(f"wrote {OUT_PATH}: {len(cells)} cells")
    for idx, cell in enumerate(cells):
        tag = (cell.metadata.get("tags") or ["?"])[0]
        first = cell.source.split("\n", 1)[0][:72]
        print(f"  [{idx:>2}] [{tag}] {first}")


if __name__ == "__main__":
    main()
