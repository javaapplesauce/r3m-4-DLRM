import numpy as np

TASK_DESCRIPTIONS = {
    "Lift": "the red cube",
    "Stack": "the red cube",
    "NutAssembly": "the nut",
    "NutAssemblySquare": "the square nut",
    "NutAssemblyRound": "the round nut",
    "PickPlace": "the object on the table",
    "PickPlaceCan": "the can",
    "PickPlaceBread": "the bread",
    "PickPlaceCereal": "the cereal box",
    "PickPlaceMilk": "the milk carton",
    "Door": "the door handle",
}


def make_env(cfg):
    """Create a robosuite environment from config."""
    import robosuite as suite

    env_name = cfg["env"]["name"]
    env = suite.make(
        env_name,
        robots=cfg["env"]["robots"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cfg["env"]["camera_name"],
        camera_heights=cfg["env"]["camera_height"],
        camera_widths=cfg["env"]["camera_width"],
        horizon=cfg["env"]["horizon"],
        reward_shaping=cfg["env"]["reward_shaping"],
    )
    return env


def get_task_description(env_name):
    return TASK_DESCRIPTIONS.get(env_name, "the target object")


def extract_obs(obs, camera_name="agentview"):
    """Extract image and proprioceptive state from robosuite observation dict.

    Returns:
        image: (3, H, W) uint8 numpy array
        proprio: (14,) float array [7 joint pos + 3 eef pos + 4 eef quat]
    """
    image = obs[f"{camera_name}_image"]
    if image.ndim == 3 and image.shape[2] == 3:
        image = image.transpose(2, 0, 1)
    image = np.flip(image, axis=1).copy()

    joint_pos = obs.get("robot0_joint_pos", np.zeros(7))
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
    proprio = np.concatenate([joint_pos, eef_pos, eef_quat]).astype(np.float32)

    return image, proprio
