# 用途：下载本地 micromamba 可执行文件，用于创建独立 OmniNav 环境。
#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${MICROMAMBA_INSTALL_DIR:-/file_system/vepfs/algorithm/intern03/.local/bin}"
MICROMAMBA_BIN="${INSTALL_DIR}/micromamba"

mkdir -p "${INSTALL_DIR}"

if [[ -x "${MICROMAMBA_BIN}" ]]; then
  echo "micromamba already exists: ${MICROMAMBA_BIN}"
  "${MICROMAMBA_BIN}" --version
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "downloading micromamba -> ${MICROMAMBA_BIN}"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "${TMP_DIR}" bin/micromamba
mv "${TMP_DIR}/bin/micromamba" "${MICROMAMBA_BIN}"
chmod +x "${MICROMAMBA_BIN}"
"${MICROMAMBA_BIN}" --version
