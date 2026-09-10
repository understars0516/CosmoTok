#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# 1. Select the Python interpreter.
#    Priority: $PYTHON_BIN override > active conda env > `cosmogrid` env > PATH python.
#    Never fall back silently to a bare base-env python (it lacks the deps).
# ---------------------------------------------------------------------------
resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "${PYTHON_BIN}"
        return 0
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        printf '%s\n' "${CONDA_PREFIX}/bin/python"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        local cosmogrid_py
        cosmogrid_py="$(conda env list 2>/dev/null | awk '$1=="cosmogrid"{print $2}')"
        if [[ -n "${cosmogrid_py}" && -x "${cosmogrid_py}/bin/python" ]]; then
            printf '%s\n' "${cosmogrid_py}/bin/python"
            return 0
        fi
    fi
    command -v python
}

PYTHON_BIN="$(resolve_python)"
echo "Using Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

# ---------------------------------------------------------------------------
# 2. Environment sanity check — fail fast with an actionable message.
# ---------------------------------------------------------------------------
if ! "${PYTHON_BIN}" - <<'PY'
import sys

required = [
    "torch", "numpy", "scipy", "einops", "jaxtyping", "jsonargparse",
    "lightning", "torchmetrics", "spdl", "intervaltree", "h5py",
    "healpy", "matplotlib",
]
missing = []
for m in required:
    try:
        __import__(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m} ({type(e).__name__}: {e})")

if missing:
    print("ERROR: missing required packages:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print(
        "\nActivate the cosmogrid env and install the dependencies:\n"
        "  conda activate cosmogrid\n"
        "  pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

import torch
if not torch.cuda.is_available():
    print(
        "WARNING: CUDA is not available; training will run on CPU (very slow).",
        file=sys.stderr,
    )
print(f"Environment OK (torch {torch.__version__}, CUDA available: {torch.cuda.is_available()})")
PY
then
    echo "Environment check failed — see the errors above." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Site-packages / PYTHONPATH setup (unchanged behaviour).
# ---------------------------------------------------------------------------
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -n "${SITE_PACKAGES}" && -w "${SITE_PACKAGES}" ]]; then
    export PYTHONNOUSERSITE=1
else
    unset PYTHONNOUSERSITE
fi
USER_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
if [[ -n "${USER_SITE}" && -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${USER_SITE}"
fi
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${SCRIPT_DIR}:${SCRIPT_DIR}/src"

CONFIG_PATH="${1:-src/imgtok/cfg-cosmo.yaml}"
shift || true

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    "${PYTHON_BIN}" src/imgtok/train.py -c "${CONFIG_PATH}" "$@"
