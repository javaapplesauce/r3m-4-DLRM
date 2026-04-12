# Concept-Aware Visual Representations for Robotic Manipulation

This project implements a hybrid vision-action stack that integrates DINOv2 dense geometric features with SAM's promptable concept segmentation for robotic manipulation policy learning via behavioral cloning.

## Architecture

```
RGB Image (518x518)
    |
    v
DINOv2 ViT-L/14 (frozen) --> Dense Feature Map F_d (37x37x1024)
    |
    v
SAM Concept Mask (frozen) --> Binary Mask M (37x37x1)
    |                          (from task description, e.g. "the red mug")
    v
Element-wise Filter: F_f = F_d * M
    |
    v
Spatial Average Pool --> Visual Embedding z_vis (1024,)
    |
    v
Concat [z_vis, z_proprio] --> (1038,)
    |
    v
3-Layer MLP (LayerNorm + ReLU) --> 6-DOF Action (dx, dy, dz, droll, dpitch, dyaw)
```

The foundation models (DINOv2, SAM) remain frozen. Only the MLP policy head is trained via behavioral cloning (MSE loss on expert demonstrations).

## Installation

```bash
pip install -e .
```

For simulation environments:
```bash
pip install robosuite
```

For concept masking (optional, requires GPU):
```bash
pip install segment-anything-2 transformers
```

## Usage

### 1. Collect Demonstrations

```bash
python scripts/collect_demos.py --env PickPlace --num-demos 50
python scripts/collect_demos.py --env Door --num-demos 50
```

### 2. Train Policy

```bash
# Train CAVR (proposed method)
python scripts/train.py --model cavr --env PickPlace --data-dir data/demos

# Train baselines
python scripts/train.py --model r3m --env PickPlace --data-dir data/demos
python scripts/train.py --model vc1 --env PickPlace --data-dir data/demos
```

### 3. Evaluate

```bash
python scripts/evaluate.py --model cavr --checkpoint checkpoints/cavr_PickPlace/best.pt --num-episodes 50
```

### 4. Run Full Baseline Comparison

```bash
python scripts/run_baselines.py --env PickPlace --data-dir data/demos
```

### 5. Run Ablation Studies

```bash
python scripts/run_ablation.py --env PickPlace --data-dir data/demos
```

Ablation variants:
- `cavr_vitl_masked` — Full CAVR (DINOv2 ViT-L/14 + SAM masking)
- `cavr_vitl_no_mask` — DINOv2 ViT-L/14 without masking
- `cavr_vitb_masked` — DINOv2 ViT-B/14 + SAM masking
- `cavr_vitb_no_mask` — DINOv2 ViT-B/14 without masking

## Project Structure

```
cavr/
    models/
        encoder.py        # DINOv2 dense feature extractor
        concept_mask.py   # SAM-based concept masking
        policy.py         # MLP policy head
        pipeline.py       # Full CAVR pipeline
        baselines.py      # R3M and VC-1 baseline wrappers
    data/
        dataset.py        # HDF5/NPZ demonstration dataset
        collector.py      # Scripted demo collection
    envs/
        robosuite_envs.py # Robosuite environment wrappers
    training/
        bc_trainer.py     # Behavioral cloning trainer
    evaluation/
        evaluator.py      # Policy evaluation (success rate)
        ablation.py       # Ablation study runner
    configs/
        default.yaml      # Default configuration
scripts/
    collect_demos.py      # Collect demonstrations
    train.py              # Train policy
    evaluate.py           # Evaluate checkpoints
    run_baselines.py      # Train + evaluate all models
    run_ablation.py       # Run ablation studies
```

## License

MIT
