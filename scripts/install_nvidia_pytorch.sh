#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  TORCH_INSTALL=<nvidia_jetson_torch_wheel_url_or_path> scripts/install_nvidia_pytorch.sh

Optional environment variables:
  FORCE_TORCH_INSTALL=1   reinstall torch even when import torch already works
  SKIP_APT=1              skip apt package install
  SKIP_PROJECT_DEPS=1     skip numpy/opencv/cuda-python/onnx install

Notes:
  - Use this on Jetson/aarch64 or inside a Jetson-compatible NVIDIA container.
  - If torch is already installed, the script verifies CUDA and does not overwrite it unless FORCE_TORCH_INSTALL=1.
  - For NVIDIA Jetson wheels, set TORCH_INSTALL from NVIDIA's PyTorch for Jetson compatibility matrix.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

python_bin="${PYTHON_BIN:-python3}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: NVIDIA Jetson PyTorch wheels are aarch64-only; current arch is $(uname -m)." >&2
  exit 1
fi

if [[ -f /etc/nv_tegra_release ]]; then
  echo "Jetson release: $(head -n 1 /etc/nv_tegra_release)"
else
  echo "WARN: /etc/nv_tegra_release not found. Continuing; this may be an NVIDIA container."
fi

if [[ "${SKIP_APT:-0}" != "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends python3-pip libopenblas-dev
  else
    echo "WARN: apt-get not found; skipping system packages."
  fi
fi

$python_bin -m pip install --upgrade pip

torch_ok=0
if [[ "${FORCE_TORCH_INSTALL:-0}" != "1" ]]; then
  if $python_bin - <<'PY'
import torch
print(f"existing torch={torch.__version__}, cuda={torch.version.cuda}")
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    torch_ok=1
  fi
fi

if [[ "$torch_ok" != "1" ]]; then
  if [[ -z "${TORCH_INSTALL:-}" ]]; then
    echo "ERROR: torch is missing or CUDA is unavailable, and TORCH_INSTALL is not set." >&2
    echo "Set TORCH_INSTALL to the NVIDIA Jetson PyTorch wheel URL/path for your JetPack version." >&2
    echo "Example:" >&2
    echo "  TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/vXX/pytorch/<torch-wheel>.whl $0" >&2
    exit 1
  fi

  $python_bin -m pip uninstall -y torch torchvision torchaudio || true
  $python_bin -m pip install --no-cache-dir "$TORCH_INSTALL"
fi

if [[ "${SKIP_PROJECT_DEPS:-0}" != "1" ]]; then
  $python_bin -m pip install --no-cache-dir \
    'numpy>=1.23.5,<2' \
    'opencv-python==4.11.0.86' \
    'cuda-python>=12,<13' \
    onnx
fi

$python_bin - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch cuda={torch.version.cuda}")
print(f"cuda available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch list={torch.cuda.get_arch_list()}")
    x = torch.ones(1, device="cuda")
    print(f"cuda tensor ok={x.item()}")
else:
    raise SystemExit("ERROR: torch installed, but CUDA is not available")
PY
