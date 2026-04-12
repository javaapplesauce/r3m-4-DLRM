# CAVR Setup Guide

Complete instructions to go from a fresh clone to running simulations.

## Prerequisites

- **OS**: Linux (recommended) or macOS (Intel or Apple Silicon)
- **Python**: 3.10 or 3.11 (3.12 has known rendering issues with MuJoCo)
- **GPU**: NVIDIA GPU with CUDA for training (CPU works but is slow)
- **Disk**: ~5 GB for model weights (DINOv2, SAM2) and demo data

## Step 1: Clone and Create Environment

```bash
git clone <this-repo-url> cavr && cd cavr

conda create -n cavr python=3.11 -y
conda activate cavr
```

If you don't use conda, any Python 3.10-3.11 virtualenv works:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

## Step 2: Install PyTorch

Install PyTorch with CUDA support matching your driver. Check https://pytorch.org/get-started/locally/ for the exact command.

**Linux with CUDA 12.x:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**macOS (CPU only):**
```bash
pip install torch torchvision
```

Verify:
```bash
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
```

## Step 3: Install CAVR and Dependencies

```bash
# Core package + simulation dependencies
pip install -e ".[all]"
```

This installs:
- `robosuite` (simulation environments)
- `mujoco` (physics engine, installed via pip — no system install needed)
- `transformers` (for Grounding DINO, used in concept masking)
- `h5py`, `pyyaml`, `tqdm`, `wandb`

If you only need the core package without simulation:
```bash
pip install -e .
```

## Step 4: Set Rendering Backend

MuJoCo needs a rendering backend for offscreen image generation.

**Linux (headless server):**
```bash
# Option A: EGL (NVIDIA GPU, no display needed — recommended)
export MUJOCO_GL=egl

# Option B: OSMesa (CPU rendering, works everywhere)
sudo apt-get install libosmesa6-dev
export MUJOCO_GL=osmesa
```

**Linux (with display):**
```bash
export MUJOCO_GL=glfw
```

**macOS:**
```bash
# No action needed — uses CGL automatically.
# If you hit issues, explicitly set:
export MUJOCO_GL=cgl
```

Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`) so it persists:
```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc  # adjust for your OS
```

## Step 5: Verify Installation

Run the full verification suite:

```bash
python scripts/verify_setup.py
```

This checks:
1. Python version is 3.10-3.11
2. PyTorch and CUDA
3. robosuite and mujoco are importable
4. An environment can be created and stepped
5. Camera observations are produced correctly
6. DINOv2 weights download and run
7. Full CAVR forward pass works
8. Training loop runs end-to-end (2-epoch smoke test)

All checks should pass. If any fail, see Troubleshooting below.

## Step 6: Collect Demonstrations

You need demonstration data before training. Two options:

### Option A: Scripted Collection (headless, no interaction)

Uses a heuristic controller. Good for testing the pipeline, but demos may be low quality for complex tasks.

```bash
# Collect 50 demos for PickPlace (the primary evaluation task)
python scripts/collect_demos.py --env PickPlace --num-demos 50

# Collect 50 demos for Door
python scripts/collect_demos.py --env Door --num-demos 50 --save-dir data/demos_door
```

Demos are saved as HDF5 to `data/demos/demos.hdf5`.

### Option B: Keyboard Teleoperation (requires display)

Higher quality demos, but requires an interactive session:

```bash
python -c "
from cavr.data.collector import collect_teleop_demos
import yaml
with open('cavr/configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['env']['name'] = 'PickPlace'
collect_teleop_demos(cfg, num_demos=50)
"
```

Controls: arrow keys (x/y), W/S (z), Q/E (gripper), Space (save demo), N (discard).

## Step 7: Train

```bash
# Train CAVR (proposed method)
python scripts/train.py --model cavr --env PickPlace --data-dir data/demos

# Train R3M baseline
python scripts/train.py --model r3m --env PickPlace --data-dir data/demos

# Train VC-1 baseline
python scripts/train.py --model vc1 --env PickPlace --data-dir data/demos
```

Training flags:
- `--epochs 200` (default)
- `--lr 1e-4` (default)
- `--batch-size 64` (reduce if OOM)
- `--wandb` to enable Weights & Biases logging

Checkpoints are saved to `checkpoints/<model>_<env>/best.pt`.

## Step 8: Evaluate

```bash
python scripts/evaluate.py \
    --model cavr \
    --checkpoint checkpoints/cavr_PickPlace/best.pt \
    --env PickPlace \
    --num-episodes 50
```

## Step 9: Run Comparisons and Ablations

Full baseline comparison (trains + evaluates all 3 models):
```bash
python scripts/run_baselines.py --env PickPlace --data-dir data/demos
```

Ablation study (ViT-B vs ViT-L, masked vs unmasked):
```bash
python scripts/run_ablation.py --env PickPlace --data-dir data/demos
```

Results are saved as JSON to `outputs/`.

---

## Troubleshooting

### `GLEW initialization error` or `Failed to initialize OpenGL`

Rendering backend mismatch. On a headless Linux server:
```bash
export MUJOCO_GL=egl    # if you have an NVIDIA GPU
export MUJOCO_GL=osmesa  # if CPU-only
```

On macOS:
```bash
export MUJOCO_GL=cgl
```

### `RuntimeError: ... Python 3.12 ...` or ctypes errors

MuJoCo's Python bindings have issues with Python 3.12. Downgrade:
```bash
conda create -n cavr python=3.11 -y && conda activate cavr
```

### `torch.hub.load` fails to download DINOv2

You may be behind a proxy or firewall. Download manually:
```bash
git clone https://github.com/facebookresearch/dinov2.git ~/.cache/torch/hub/facebookresearch_dinov2_main
```

### Out of Memory during training

Reduce batch size:
```bash
python scripts/train.py --batch-size 16
```

Or use the smaller backbone:
```yaml
# In cavr/configs/default.yaml
encoder:
  backbone: "dinov2_vitb14"  # 768-dim instead of 1024-dim
```

### robosuite environment hangs on `env.reset()`

This usually means the renderer failed silently. Verify MuJoCo works standalone:
```bash
python -c "import mujoco; print(mujoco.__version__)"
python -c "
import robosuite as suite
env = suite.make('Lift', robots='Panda', has_renderer=False, has_offscreen_renderer=False)
env.reset()
env.close()
print('OK')
"
```

If the above works but adding `has_offscreen_renderer=True` hangs, it's the GL backend — see the first troubleshooting item.

### Apple Silicon: `illegal hardware instruction`

Ensure you're using the arm64 versions of all packages:
```bash
python -c "import platform; print(platform.machine())"  # should print arm64
```

If it prints `x86_64`, you're running under Rosetta. Create a native env:
```bash
arch -arm64 conda create -n cavr python=3.11 -y
```

---

## Full Pipeline Summary

```
1. conda create + activate
2. pip install torch torchvision
3. pip install -e ".[all]"
4. export MUJOCO_GL=egl  (Linux) or skip (macOS)
5. python scripts/verify_setup.py
6. python scripts/collect_demos.py --env PickPlace --num-demos 50
7. python scripts/train.py --model cavr --env PickPlace
8. python scripts/evaluate.py --model cavr --checkpoint checkpoints/cavr_PickPlace/best.pt
9. python scripts/run_baselines.py --env PickPlace
10. python scripts/run_ablation.py --env PickPlace
```
