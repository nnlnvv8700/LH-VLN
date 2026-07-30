# 用途：下载 OmniNav slow-fast checkpoint 到 NAS，避免模型大文件进入源码目录。
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python}"
HELPER_VENV="${OMNINAV_DOWNLOAD_VENV:-/file_system/vepfs/algorithm/intern03/.venvs/modelscope_download}"
MODEL_ID="${OMNINAV_MODEL_ID:-chongchongjj/OmniNav_Slowfast}"
LOCAL_DIR="${OMNINAV_MODEL_DIR:-/file_system/nas/algorithm/Intern03/models/OmniNav/OmniNav_Slowfast}"

mkdir -p "$(dirname "${HELPER_VENV}")" "${LOCAL_DIR}"

if [[ ! -x "${HELPER_VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${HELPER_VENV}"
  "${HELPER_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${HELPER_VENV}/bin/python" -m pip install modelscope
fi

"${HELPER_VENV}/bin/python" - <<PYCODE
from modelscope.hub.snapshot_download import snapshot_download
model_id = "${MODEL_ID}"
local_dir = "${LOCAL_DIR}"
print(f"downloading {model_id} -> {local_dir}")
snapshot_download(model_id, local_dir=local_dir)
print("download complete")
PYCODE
