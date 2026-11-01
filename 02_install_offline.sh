cat << 'EOF' > ~/02_install_offline.sh
#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_FILE="$HOME/qt5_ubuntu24_offline.tar.gz"
INSTALL_DIR="/opt/offline_repo"
REPO_LIST="/etc/apt/sources.list.d/offline_local.list"
BACKUP_DIR="/etc/apt/sources.backup.$$"

# Fail-Safe 1: Ensure root execution
if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: Run this script with sudo: sudo ./02_install_offline.sh" >&2
    exit 1
fi

# Fail-Safe 2: Distro and ABI verification
if [[ ! -f /etc/os-release ]]; then
    echo "ERROR: /etc/os-release missing. Cannot verify OS." >&2
    exit 1
fi
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]] || [[ "${VERSION_ID:-}" != "24.04" ]]; then
    echo "FATAL: Detected ${ID:-unknown} ${VERSION_ID:-unknown}. Expected Ubuntu 24.04." >&2
    echo "Halting immediately to prevent system/kernel library mismatch." >&2
    exit 1
fi

ARCH=$(uname -m)
if [[ "${ARCH}" != "x86_64" ]]; then
    echo "FATAL: Detected architecture ${ARCH}. Expected x86_64." >&2
    exit 1
fi

# Fail-Safe 3: Verify archive existence
if [[ ! -f "${ARCHIVE_FILE}" ]]; then
    echo "ERROR: '${ARCHIVE_FILE}' not found." >&2
    exit 1
fi

# Fail-Safe 4: Check minimum 4 GB storage
FREE_KB=$(df --output=avail / | tail -n1)
if [[ "${FREE_KB}" -lt 4194304 ]]; then
    echo "ERROR: Less than 4 GB free space on /. Aborting." >&2
    exit 1
fi

# Fail-Safe 5: Cleanup & source list recovery trap
cleanup() {
    echo "--> Restoring repository configurations..."
    rm -f "${REPO_LIST}"
    if [[ -d "${BACKUP_DIR}" ]]; then
        mv "${BACKUP_DIR}"/* /etc/apt/sources.list.d/ 2>/dev/null || true
        rm -rf "${BACKUP_DIR}"
    fi
}
trap cleanup EXIT INT TERM

echo "=== [1/6] Unpacking Repository Archive ==="
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
tar -xzvf "${ARCHIVE_FILE}" -C /opt

if [[ ! -f "${INSTALL_DIR}/debs/Packages.gz" ]]; then
    echo "ERROR: Repository metadata (Packages.gz) missing from archive." >&2
    exit 1
fi

echo "=== [2/6] Configuring Temporary Offline APT Channel ==="
mkdir -p "${BACKUP_DIR}"
find /etc/apt/sources.list.d/ -maxdepth 1 -type f \( -name "*.sources" -o -name "*.list" \) | while read -r src; do
    mv "$src" "${BACKUP_DIR}/"
done

echo "deb [trusted=yes] file:${INSTALL_DIR}/debs ./" > "${REPO_LIST}"

echo "=== [3/6] Updating APT Cache from Local Files ==="
apt-get update -o Dir::Etc::sourcelist="${REPO_LIST}" -o Dir::Etc::sourceparts="-"

mapfile -t TARGET_PACKAGES < "${INSTALL_DIR}/target_packages.txt"

echo "=== [4/6] Executing Dry-Run Simulation (Fail-Safe Check) ==="
if ! apt-get install --dry-run --no-download -y "${TARGET_PACKAGES[@]}"; then
    echo "FATAL: Simulation failed. Package conflict or missing dependency detected." >&2
    echo "No changes were made to your system." >&2
    exit 1
fi

echo "=== [5/6] Installing Packages ==="
apt-get install --no-download -y "${TARGET_PACKAGES[@]}"

echo "=== [6/6] Verifying Toolchain ==="
echo "GCC:       $(gcc --version | head -n1)"
echo "G++:       $(g++ --version | head -n1)"
echo "CMake:     $(cmake --version | head -n1)"
echo "QMake:     $(qmake -v | grep 'Using Qt version' || true)"
echo "QtCreator: $(which qtcreator)"

echo "=== ALL PACKAGES INSTALLED SUCCESSFULLY ==="
EOF