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
            "sam-2 @ git+https://github.com/facebookresearch/sam2.git",
        ],
        "baselines": [
            "r3m @ git+https://github.com/facebookresearch/r3m.git",
            "timm>=0.9",
        ],
        "logging": ["wandb"],
        # "all" excludes sam-2 on purpose: it's a CUDA-built extension that
        # often fails on fresh environments. ConceptMasker falls back to an
        # all-ones mask with a RuntimeWarning when sam2 is absent. Install
        # the [masking] extra explicitly if you need real concept masking.
        "all": [
            "robosuite>=1.4",
            "mujoco>=3.0",
            "transformers>=4.30",
            "r3m @ git+https://github.com/facebookresearch/r3m.git",
            "timm>=0.9",
            "wandb",
        ],
    },
)
