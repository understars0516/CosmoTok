#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

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

CONFIG_PATH="${1:-src/imgtok/cfg-cosmo-adv.yaml}"
shift || true

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-8}" \
"${PYTHON_BIN}" src/imgtok/train.py -c "${CONFIG_PATH}" "$@"
