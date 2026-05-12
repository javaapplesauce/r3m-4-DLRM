# CAVR: Concept-Aware Visual Representations for Robotic Manipulation
Final project, COMS 6998: Deep Learning for Robot Manipulation, Columbia University. Richard Li and Luke Yuan.

## Summary
CAVR combines a frozen DINOv2 ViT-L/14 dense feature map with a Grounding-DINO → SAM 2 concept mask, derived from the natural-language task description, to produce a task-focused visual embedding for behavioral cloning. We benchmark CAVR against R3M and VC-1 on the robosuite Lift task and ablate the masking step. Across 20 evaluation rollouts, CAVR outperforms both baselines and ablating the concept mask drops success rate to the level of the unmasked baselines, indicating that the localization signal — not just the DINOv2 features — carries the gain.

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

Only the MLP head is trained. DINOv2, Grounding-DINO, and SAM 2 remain frozen.

## Results

| Method                       | Lift success |
|------------------------------|--------------|
| CAVR (DINOv2 + GD + SAM 2)   | 17/20 (85%)  |
| R3M                          | 16/20 (80%)  |
| VC-1                         | 14/20 (70%)  |
| CAVR, no masking (ablation)  | 16/20 (80%)  |

N=20 rollouts, single seed; see paper for confidence intervals and statistical caveats.

## Setup

```bash
pip install -e .
```

For simulation (robosuite + mujoco):
```bash
pip install -e ".[all]"
```

For concept masking (Grounding-DINO via `transformers` plus SAM 2 from source) and the SAM 2 checkpoint:
```bash
pip install transformers timm
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

If SAM 2 or its checkpoint is unavailable, `ConceptMasker` falls back to an all-ones mask and emits a `RuntimeWarning`; training still runs but masking becomes a no-op.

## Reproducing the results

```bash
# 1. Collect 50 scripted Lift demonstrations into data/demos_Lift/demos.hdf5
python scripts/collect_demos.py --env Lift --num-demos 50 --save-dir data/demos_Lift

# 2. Precompute Grounding-DINO + SAM 2 masks for every demo frame
python scripts/precompute_masks.py --env Lift --data-dir data/demos_Lift

# 3. Train + evaluate CAVR, R3M, and VC-1 on Lift
python scripts/run_baselines.py --env Lift --data-dir data/demos_Lift

# 4. Run the masking ablation (ViT-L masked vs. ViT-L no-mask, etc.)
python scripts/run_ablation.py --env Lift --data-dir data/demos_Lift
```

For the end-to-end Colab path (clone → install → collect → train sweep → figures), open `colab/cavr_e2e.ipynb` in a Colab A100 runtime.

## Project structure

```
cavr/
    models/
        encoder.py        # DINOv2 dense feature extractor
        concept_mask.py   # Grounding-DINO + SAM 2 concept masking
        policy.py         # MLP policy head
        pipeline.py       # Full CAVR pipeline
        baselines.py      # R3M and VC-1 baseline wrappers
    data/
        dataset.py        # HDF5 demonstration dataset
        collector.py      # Scripted demo collection
    envs/
        robosuite_envs.py # Robosuite environment wrappers
    training/
        bc_trainer.py     # Behavioral cloning trainer
    evaluation/
        evaluator.py      # Policy evaluation (success rate)
        ablation.py       # Ablation study runner
    utils/
        io.py             # Atomic JSON/CSV writers
        reporting.py      # Resume status + mask spot-check panel
        runs.py           # train_and_eval orchestrator
    configs/
        default.yaml      # Default configuration
scripts/
    collect_demos.py      # Collect demonstrations
    precompute_masks.py   # Cache Grounding-DINO + SAM 2 masks
    train.py              # Train a single policy
    evaluate.py           # Evaluate a checkpoint
    run_baselines.py      # Train + evaluate CAVR / R3M / VC-1
    run_ablation.py       # Train + evaluate CAVR ablation variants
colab/
    cavr_e2e.ipynb        # End-to-end Colab notebook
```

## Paper
See `paper/paper.pdf` for the full writeup.

## License
MIT.
