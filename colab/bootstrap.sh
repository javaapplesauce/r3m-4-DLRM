#!/usr/bin/env bash
# One-shot Colab bootstrap for CAVR.
# Usage (from the first cell of a Colab notebook):
#   !git clone https://github.com/<user>/r3m-4-DLRM.git /content/cavr
#   !bash /content/cavr/colab/bootstrap.sh
set -euo pipefail

REPO_DIR="${1:-/content/cavr}"
cd "$REPO_DIR"

echo "[cavr] GPU check"
nvidia-smi || echo "WARNING: no GPU detected; switch Colab runtime to A100."

echo "[cavr] System deps for MuJoCo / EGL"
apt-get -qq update
apt-get -qq install -y libegl1 libgl1 libosmesa6 libglfw3 patchelf > /dev/null

echo "[cavr] Python deps"
pip install -q --upgrade pip
# Torch is preinstalled on Colab with matching CUDA — don't clobber it.
pip install -q -e ".[all]"

echo "[cavr] SAM2 checkpoint (skip if already downloaded)"
if [ ! -f sam2_hiera_large.pt ]; then
  wget -q --show-progress \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    -O sam2_hiera_large.pt || echo "WARNING: SAM2 ckpt download failed; masking will fall back to all-ones."
fi

echo "[cavr] Rendering backend: EGL (A100 on Colab has NVIDIA driver)"
# Persist for subsequent cells via /etc/environment + os.environ in the notebook.
grep -q MUJOCO_GL /etc/environment || echo 'MUJOCO_GL=egl' >> /etc/environment
grep -q PYOPENGL_PLATFORM /etc/environment || echo 'PYOPENGL_PLATFORM=egl' >> /etc/environment

echo "[cavr] Bootstrap complete."
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
