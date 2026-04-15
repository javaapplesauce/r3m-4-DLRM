from setuptools import setup, find_packages

setup(
    name="cavr",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    description="Concept-Aware Visual Representations for Robotic Manipulation",
    author="Richard Li, Luke Yuan",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "pyyaml",
        "tqdm",
        "h5py",
    ],
    extras_require={
        "sim": ["robosuite>=1.4", "mujoco>=3.0"],
        "masking": [
            "transformers>=4.30",
            "sam2 @ git+https://github.com/facebookresearch/sam2.git",
        ],
        "baselines": [
            "r3m @ git+https://github.com/facebookresearch/r3m.git",
            "timm>=0.9",
        ],
        "logging": ["wandb"],
        "all": [
            "robosuite>=1.4",
            "mujoco>=3.0",
            "transformers>=4.30",
            "sam2 @ git+https://github.com/facebookresearch/sam2.git",
            "r3m @ git+https://github.com/facebookresearch/r3m.git",
            "timm>=0.9",
            "wandb",
        ],
    },
)
