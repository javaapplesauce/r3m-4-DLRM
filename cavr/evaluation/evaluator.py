import numpy as np
import torch
from tqdm import trange

from cavr.envs.robosuite_envs import make_env, extract_obs, get_task_description


# Object-position keys the gripper heuristic looks for, in priority order.
# Mirrors cavr.data.collector.OBJECT_POS_KEYS so eval and collection share
# the same notion of "the target object".
_OBJECT_POS_KEYS = (
    "cube_pos",
    "Can_pos", "Milk_pos", "Bread_pos", "Cereal_pos",
    "handle_pos", "door_pos",
)


def _target_pos_from_obs(obs):
    for key in _OBJECT_POS_KEYS:
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
    for k, v in obs.items():
        if k.endswith("_pos") and not k.startswith("robot"):
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape == (3,):
                return arr
    return None


def _gripper_heuristic(obs):
    """State-based gripper command for eval-time policies trained with 6-DOF
    actions only. The collector saved action6 (no gripper channel), so the
    policy can't predict the gripper; we infer it from proximity to the
    target object — close when the eef is roughly on top of the target,
    open otherwise. This matches the collector's phase machine in spirit.

    Returns a scalar in [-1, 1] suitable for the env's 7th action channel.
    """
    eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    target = _target_pos_from_obs(obs)
    if target is None:
        return 0.0
    dxy = float(np.linalg.norm(eef[:2] - target[:2]))
    dz = float(eef[2] - target[2])
    # "Close enough to grasp": xy within ~3 cm, vertically within 4 cm above
    # the target (so we also close during the lift phase, not only the
    # instant of contact).
    if dxy < 0.03 and -0.02 < dz < 0.04:
        return 1.0
    if dxy < 0.05 and dz < 0.10:
        # Approaching — stay open to make sure we don't snag.
        return -1.0
    return -1.0


class PolicyEvaluator:
    """Evaluates a trained policy in simulation by measuring success rate."""

    def __init__(self, model, cfg, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cfg = cfg

    @torch.no_grad()
    def evaluate(self, num_episodes=None):
        num_episodes = num_episodes or self.cfg["evaluation"]["num_episodes"]
        env = make_env(self.cfg)
        camera = self.cfg["env"]["camera_name"]
        task_desc = get_task_description(self.cfg["env"]["name"])
        horizon = self.cfg["env"]["horizon"]

        successes = 0
        episode_lengths = []
        episode_returns = []

        for ep in trange(num_episodes, desc="Evaluating"):
            obs = env.reset()
            total_reward = 0.0
            success = False

            for t in range(horizon):
                image, proprio = extract_obs(obs, camera)
                image_t = torch.from_numpy(image).unsqueeze(0).to(self.device)
                proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)

                action = self.model(image_t, proprio_t, task_desc)
                action_np = action.squeeze(0).cpu().numpy()

                # The policy was trained on 6-DOF actions only (collector
                # never stored the gripper channel). At eval, fill the
                # gripper slot via a state-based heuristic so the policy
                # actually has a chance of grasping.
                full_action = np.zeros(env.action_dim)
                full_action[:len(action_np)] = action_np
                if env.action_dim > len(action_np):
                    full_action[len(action_np)] = _gripper_heuristic(obs)

                obs, reward, done, info = env.step(full_action)
                total_reward += reward

                if env._check_success():
                    success = True
                    episode_lengths.append(t + 1)
                    break

            if not success:
                episode_lengths.append(horizon)
            successes += int(success)
            episode_returns.append(total_reward)

        env.close()

        results = {
            "success_rate": successes / num_episodes,
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "num_episodes": num_episodes,
        }

        print(f"\n{'='*50}")
        print(f"Task: {self.cfg['env']['name']}")
        print(f"Success Rate: {results['success_rate']:.2%} ({successes}/{num_episodes})")
        print(f"Mean Return:  {results['mean_return']:.2f} +/- {results['std_return']:.2f}")
        print(f"Mean Length:  {results['mean_length']:.1f}")
        print(f"{'='*50}\n")

        return results
