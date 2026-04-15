"""Scripted expert demonstration collection for robosuite tasks.

The scripted policy is a simple heuristic operating in the robot end-effector
frame. It is *not* a high-quality expert — it exists to produce enough
successful episodes to unblock training and evaluation of CAVR. Episodes that
do not reach task success (by robosuite's own success criterion) are discarded.

Action convention written to disk (cfg["policy"]["action_dim"] = 6):
    [dx, dy, dz, droll, dpitch, dyaw]
Gripper is commanded by the collector via a phase-based heuristic but NOT stored.
At evaluation time, gripper control must be supplied by a similar heuristic.
"""
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from cavr.envs.robosuite_envs import make_env, extract_obs


def _target_position(env, env_name):
    """Return a 3-vec world-frame target to reach for the current env state."""
    try:
        if env_name.startswith("PickPlace"):
            obj = env.objects[0] if hasattr(env, "objects") and env.objects else None
            if obj is not None and hasattr(env, "sim"):
                return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)])
        if env_name == "Door":
            handle_id = env.sim.model.site_name2id("door_handle")
            return np.array(env.sim.data.site_xpos[handle_id])
    except Exception:
        pass
    # Fallback: cube-like object at robosuite default
    return np.array([0.0, 0.0, 0.9])


def _scripted_step(env, env_name, phase):
    """Return (action_6d, gripper_cmd, next_phase) for a heuristic expert."""
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")] \
        if "gripper0_grip_site" in env.sim.model.site_names \
        else np.array(env._eef_xpos) if hasattr(env, "_eef_xpos") else np.zeros(3)

    target = _target_position(env, env_name)
    delta = target - eef_pos
    dist = np.linalg.norm(delta[:2])

    gripper_cmd = -1.0  # open
    if phase == "approach":
        direction = np.concatenate([delta[:2], [0.05]])
        if dist < 0.02:
            phase = "descend"
    elif phase == "descend":
        direction = np.array([0.0, 0.0, -0.05])
        if delta[2] > -0.005:
            phase = "grasp"
    elif phase == "grasp":
        direction = np.zeros(3)
        gripper_cmd = 1.0
        phase = "lift"
    elif phase == "lift":
        direction = np.array([0.0, 0.0, 0.08])
        gripper_cmd = 1.0
        if eef_pos[2] > 1.05:
            phase = "carry"
    else:  # carry/hold
        direction = np.zeros(3)
        gripper_cmd = 1.0

    direction = np.clip(direction, -0.1, 0.1)
    action6 = np.concatenate([direction, np.zeros(3)]).astype(np.float32)
    return action6, gripper_cmd, phase


def _rollout_one(env, cfg, max_retries=5):
    """Run the scripted policy until success or horizon. Returns None on failure."""
    env_name = cfg["env"]["name"]
    cam = cfg["env"]["camera_name"]
    horizon = cfg["env"]["horizon"]

    obs = env.reset()
    phase = "approach"
    images, proprios, actions = [], [], []

    for _ in range(horizon):
        img, proprio = extract_obs(obs, camera_name=cam)
        action6, gripper, phase = _scripted_step(env, env_name, phase)
        full_action = np.concatenate([action6, [gripper]]).astype(np.float32)
        # robosuite default OSC_POSE controller expects 7-dim [pose6, gripper].
        # If the controller action dim differs, pad/truncate.
        ctrl_dim = env.action_dim
        if full_action.shape[0] < ctrl_dim:
            full_action = np.pad(full_action, (0, ctrl_dim - full_action.shape[0]))
        elif full_action.shape[0] > ctrl_dim:
            full_action = full_action[:ctrl_dim]

        images.append(img)
        proprios.append(proprio)
        actions.append(action6)

        obs, _reward, done, _info = env.step(full_action)
        if env._check_success():
            return np.stack(images), np.stack(proprios), np.stack(actions)
        if done:
            return None
    return None


def collect_scripted_demos(cfg):
    """Collect N successful scripted demos and write them to HDF5."""
    save_dir = Path(cfg["data"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "demos.hdf5"

    env = make_env(cfg)
    num_demos = int(cfg["data"]["num_demos"])

    with h5py.File(out, "w") as f:
        f.attrs["env_name"] = cfg["env"]["name"]
        f.attrs["action_dim"] = cfg["policy"]["action_dim"]
        f.attrs["proprio_dim"] = cfg["policy"]["proprio_dim"]
        f.attrs["image_size"] = cfg["env"]["camera_height"]

        saved = 0
        attempts = 0
        max_attempts = num_demos * 10
        pbar = tqdm(total=num_demos, desc=f"collecting {cfg['env']['name']}")
        while saved < num_demos and attempts < max_attempts:
            attempts += 1
            result = _rollout_one(env, cfg)
            if result is None:
                continue
            imgs, props, acts = result
            g = f.create_group(f"demo_{saved}")
            g.create_dataset("images", data=imgs, compression="gzip", compression_opts=4)
            g.create_dataset("proprio", data=props)
            g.create_dataset("actions", data=acts)
            saved += 1
            pbar.update(1)
        pbar.close()

    env.close()
    print(f"[collector] wrote {saved}/{num_demos} successful demos to {out} "
          f"(attempts: {attempts})")
    if saved == 0:
        raise RuntimeError(
            "Scripted collector failed to produce any successful episode. "
            "The heuristic expert is task-dependent — tune _scripted_step in "
            "cavr/data/collector.py for your env, or use teleop collection."
        )


def collect_teleop_demos(cfg, num_demos=None):
    """Keyboard-teleoperated collection. Requires a display; Colab cannot run this."""
    raise NotImplementedError(
        "Teleop collection is not supported in headless environments (e.g. Colab). "
        "Use collect_scripted_demos or record demos on a local machine with a display."
    )
