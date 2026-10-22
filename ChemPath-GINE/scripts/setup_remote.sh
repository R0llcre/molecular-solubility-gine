#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-chempath-gine}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/ChemPath-GINE}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
INSTALL_PYG_EXTENSIONS="${INSTALL_PYG_EXTENSIONS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-auto}"

echo "[setup] project: ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "[setup] installing micromamba"
  mkdir -p "$HOME/.local/bin" "$HOME/micromamba"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$HOME/.local/bin" --strip-components=1 bin/micromamba
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"

if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] creating ${ENV_NAME}"
  micromamba create -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" \
    rdkit pandas numpy scipy scikit-learn pyyaml tqdm pytest pip
fi

micromamba activate "${ENV_NAME}"
python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] detected NVIDIA GPU"
  nvidia-smi || true
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_TAG}}"
else
  echo "[setup] no NVIDIA GPU detected; using CPU PyTorch"
  TORCH_CUDA_TAG="cpu"
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
fi

echo "[setup] installing PyTorch from ${TORCH_INDEX_URL}"
python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
python -m pip install --upgrade torch_geometric

if [[ "${INSTALL_PYG_EXTENSIONS}" == "1" ]]; then
  PYG_WHEEL_URL="$(python - <<'PY'
import torch

version = torch.__version__.split("+", 1)[0].split(".")
major_minor = ".".join(version[:2])
if major_minor in {"2.10", "2.11"}:
    torch_tag = "2.10.0"
else:
    torch_tag = f"{major_minor}.0"
cuda = torch.version.cuda
cuda_tag = "cpu" if cuda is None else "cu" + cuda.replace(".", "")
print(f"https://data.pyg.org/whl/torch-{torch_tag}+{cuda_tag}.html")
PY
)"
  echo "[setup] installing optional PyG extensions from ${PYG_WHEEL_URL}"
  python -m pip install --upgrade pyg_lib torch_scatter torch_sparse -f "${PYG_WHEEL_URL}" || {
    echo "[setup] optional PyG extensions failed; continuing with torch_geometric-only install"
  }
fi

python -m pip install -e .

echo "[setup] python: $(which python)"
python - <<'PY'
import torch, torch_geometric, rdkit, pandas
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
print("torch_geometric", torch_geometric.__version__)
print("rdkit", rdkit.__version__)
print("pandas", pandas.__version__)
PY

if [[ "${REQUIRE_CUDA}" == "1" ]] || { [[ "${REQUIRE_CUDA}" == "auto" ]] && command -v nvidia-smi >/dev/null 2>&1; }; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required on this VM but torch.cuda.is_available() is False")
PY
fi

echo "[setup] running tests"
python -m pytest -q

echo "[setup] done"
