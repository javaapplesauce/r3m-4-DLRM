import numpy as np
import torch
from tqdm import trange

from cavr.envs.robosuite_envs import make_env, extract_obs, get_task_description


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

                full_action = np.zeros(env.action_dim)
                full_action[:len(action_np)] = action_np

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
