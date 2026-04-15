"""Verify that the CAVR environment is correctly installed.

Run this after setup to confirm all components work:
    python scripts/verify_setup.py

Checks:
  1. Python version
  2. Core dependencies (torch, torchvision, numpy, yaml, h5py)
  3. CUDA availability
  4. Robosuite + MuJoCo installation
  5. Robosuite environment creation and rendering
  6. Observation extraction
  7. DINOv2 encoder loading
  8. CAVR model forward pass
  9. Dataset and training loop (smoke test)
"""
import sys
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
results = []


def check(name, fn):
    try:
        msg = fn()
        results.append((name, True, msg))
        print(f"  [{PASS}] {name}: {msg}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  [{FAIL}] {name}: {e}")
        traceback.print_exc()


def warn(name, msg):
    results.append((name, None, msg))
    print(f"  [{WARN}] {name}: {msg}")


def check_python():
    v = sys.version_info
    assert v.major == 3 and v.minor in (10, 11, 12), \
        f"Python 3.10, 3.11, or 3.12 required, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "cpu"
    return f"{torch.__version__}, device={device}"


def check_deps():
    import torchvision
    import numpy
    import yaml
    import h5py
    return (
        f"torchvision={torchvision.__version__}, "
        f"numpy={numpy.__version__}, h5py={h5py.__version__}"
    )


def check_robosuite():
    import robosuite
    return f"robosuite={robosuite.__version__}"


def check_mujoco():
    import mujoco
    return f"mujoco={mujoco.__version__}"


def check_env_creation():
    import robosuite as suite
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        horizon=50,
    )
    obs = env.reset()
    assert "agentview_image" in obs, f"Missing camera obs. Keys: {list(obs.keys())}"
    img = obs["agentview_image"]
    assert img.shape == (84, 84, 3), f"Unexpected image shape: {img.shape}"

    action = env.action_spec[0] * 0
    obs2, reward, done, info = env.step(action)
    env.close()
    return f"Lift env OK, image={img.shape}, action_dim={len(action)}"


def check_obs_extraction():
    import robosuite as suite
    import numpy as np
    from cavr.envs.robosuite_envs import extract_obs

    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        horizon=50,
    )
    obs = env.reset()
    image, proprio = extract_obs(obs, "agentview")
    env.close()
    assert image.shape == (3, 84, 84), f"Bad image shape: {image.shape}"
    assert proprio.shape == (14,), f"Bad proprio shape: {proprio.shape}"
    assert image.dtype == np.uint8, f"Bad image dtype: {image.dtype}"
    return f"image={image.shape} ({image.dtype}), proprio={proprio.shape}"


def check_dinov2():
    import torch
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model.forward_features(dummy)["x_norm_patchtokens"]
    assert out.shape == (1, 256, 768), f"Unexpected shape: {out.shape}"
    del model
    return f"dinov2_vitb14 loaded, output={out.shape}"


def check_cavr_forward():
    import torch
    import yaml
    from cavr.models.pipeline import CAVR

    with open("cavr/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["encoder"]["backbone"] = "dinov2_vitb14"
    cfg["masking"]["enabled"] = False

    model = CAVR(cfg)
    model.eval()

    B = 2
    images = torch.randint(0, 255, (B, 3, 518, 518), dtype=torch.uint8)
    proprio = torch.randn(B, 14)

    with torch.no_grad():
        action = model(images.float(), proprio)
    assert action.shape == (B, 6), f"Unexpected action shape: {action.shape}"
    del model
    return f"CAVR forward pass OK, action={action.shape}"


def check_training_smoke():
    import torch
    import yaml
    import tempfile
    import os
    import numpy as np
    import h5py
    from cavr.models.pipeline import CAVR
    from cavr.data.dataset import DemoDataset
    from cavr.training.bc_trainer import BCTrainer

    with open("cavr/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["encoder"]["backbone"] = "dinov2_vitb14"
    cfg["masking"]["enabled"] = False
    cfg["training"]["num_epochs"] = 2
    cfg["training"]["eval_freq"] = 1
    cfg["training"]["batch_size"] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = os.path.join(tmpdir, "demos.hdf5")
        N = 20
        with h5py.File(demo_path, "w") as f:
            g = f.create_group("demo_0")
            g.create_dataset(
                "images",
                data=np.random.randint(0, 255, (N, 3, 518, 518), dtype=np.uint8),
            )
            g.create_dataset("proprio", data=np.random.randn(N, 14).astype(np.float32))
            g.create_dataset("actions", data=np.random.randn(N, 6).astype(np.float32))

        cfg["training"]["checkpoint_dir"] = os.path.join(tmpdir, "ckpts")
        dataset = DemoDataset(tmpdir)
        assert len(dataset) == N, f"Dataset has {len(dataset)} items, expected {N}"

        model = CAVR(cfg)
        trainer = BCTrainer(model, cfg, device="cpu")
        val_loss = trainer.train(dataset, task_description="test object")

        ckpt_path = os.path.join(tmpdir, "ckpts", "best.pt")
        assert os.path.exists(ckpt_path), "Checkpoint not saved"

    return f"2-epoch training OK, val_loss={val_loss:.4f}"


def main():
    print("\n" + "=" * 60)
    print("CAVR Environment Verification")
    print("=" * 60 + "\n")

    print("Core dependencies:")
    check("Python version", check_python)
    check("PyTorch", check_torch)
    check("Other deps", check_deps)

    print("\nSimulation:")
    check("robosuite", check_robosuite)
    check("mujoco", check_mujoco)
    check("Environment creation", check_env_creation)
    check("Observation extraction", check_obs_extraction)

    print("\nModels:")
    check("DINOv2 download", check_dinov2)
    check("CAVR forward pass", check_cavr_forward)

    print("\nTraining:")
    check("Training smoke test", check_training_smoke)

    passed = sum(1 for _, ok, _ in results if ok is True)
    failed = sum(1 for _, ok, _ in results if ok is False)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(results)} total")
    if failed == 0:
        print("All checks passed. Environment is ready.")
    else:
        print("Some checks failed. See errors above.")
    print(f"{'=' * 60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
