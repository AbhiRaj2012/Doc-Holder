#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
QT_VERSION="6.8.0"
OUT_DIR="$(pwd)/Ubuntu_Qt_Offline"
DEB_DIR="${OUT_DIR}/ubuntu_debs"
WORK_DIR="/tmp/qt_offline_staging_$$"
VENV_DIR="${WORK_DIR}/venv"
SDK_DIR="${WORK_DIR}/Qt_SDK"

MIRROR_PRIMARY="https://ftp.jaist.ac.jp/pub/qtproject"
MIRROR_FALLBACK="https://mirrors.dotsrc.org/qtproject"

echo "=================================================================="
echo " Building Complete Offline Ubuntu x86_64 Qt ${QT_VERSION} + C++ Bundle"
echo " Output Folder : ${OUT_DIR}"
echo " Staging Folder: ${WORK_DIR} (Native Linux FS for symlinks)"
echo "=================================================================="

mkdir -p "${OUT_DIR}" "${DEB_DIR}" "${WORK_DIR}"

cleanup() {
    echo "Cleaning up temporary staging directory ${WORK_DIR}..."
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# [1/5] Prepare Host Downloader Tools & Isolated Python Venv for aqtinstall
# ------------------------------------------------------------------------------
echo ""
echo "[1/5] Installing downloader prerequisites (python3-venv, curl, p7zip)..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip curl tar p7zip-full apt-rdepends || \
sudo apt-get install -y python3 python3-venv python3-pip curl tar p7zip-full

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel >/dev/null
"${VENV_DIR}/bin/pip" install aqtinstall >/dev/null
AQT="${VENV_DIR}/bin/aqt"

# Helper function: Runs aqt against JAIST mirror, falls back to Dotsrc if needed
run_aqt() {
    if ! "${AQT}" "$@" -b "${MIRROR_PRIMARY}" --timeout 60; then
        echo "Primary mirror hiccuped. Retrying with fallback mirror (${MIRROR_FALLBACK})..."
        "${AQT}" "$@" -b "${MIRROR_FALLBACK}" --timeout 60
    fi
}

# ------------------------------------------------------------------------------
# [2/5] Download Full Ubuntu C++ Compiler (g++) & X11/OpenGL .deb Dependency Tree
# ------------------------------------------------------------------------------
echo ""
echo "[2/5] Downloading Ubuntu C++ toolchain (g++, gdb, make, cmake) & X11/OpenGL .debs..."

PKGS=(
    build-essential g++ gcc gdb make cmake ninja-build pkg-config
    libgl1-mesa-dev libglu1-mesa-dev
    libxcb-cursor0 libxcb-cursor-dev libxcb-xinerama0 libxcb-icccm4
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0
    libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xkb1
    libxkbcommon-x11-0 libxkbcommon-dev libfontconfig1-dev libfreetype6-dev
    libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev libxi-dev
    libxrender-dev libdbus-1-3
)

# Trick APT with an empty status file so it downloads ALL .debs even if already installed
EMPTY_STATUS="${WORK_DIR}/empty_dpkg_status"
touch "${EMPTY_STATUS}"

sudo apt-get install --download-only -y \
    -o APT::Immediate-Configure=false \
    -o Dir::State::status="${EMPTY_STATUS}" \
    -o Dir::Cache::archives="${DEB_DIR}" \
    "${PKGS[@]}" || {
    echo "Fallback: Downloading recursive package list via apt-cache..."
    cd "${DEB_DIR}"
    apt-cache depends --recurse --no-recommends --no-suggests \
        --no-conflicts --no-breaks --no-replaces --no-enhances \
        --no-pre-depends "${PKGS[@]}" | grep "^\w" | sort -u | \
        xargs -r -n 10 bash -c 'for p in "$@"; do apt-get download "$p" 2>/dev/null || true; done' _
    cd - >/dev/null
}

sudo rm -f "${DEB_DIR}/lock" "${DEB_DIR}"/partial/* 2>/dev/null || true
sudo rmdir "${DEB_DIR}/partial" 2>/dev/null || true
sudo chown -R "$(id -u):$(id -g)" "${DEB_DIR}"

# ------------------------------------------------------------------------------
# [3/5] Download Qt 6.8.0 Linux gcc_64 SDK, Qt Creator, CMake & Ninja
# ------------------------------------------------------------------------------
echo ""
echo "[3/5] Downloading Qt ${QT_VERSION} Linux SDK, Qt Creator, CMake & Ninja..."
mkdir -p "${SDK_DIR}"

# 1. Qt 6.8.0 Desktop SDK (gcc_64)
run_aqt install-qt linux desktop "${QT_VERSION}" linux_gcc_64 -O "${SDK_DIR}" || \
run_aqt install-qt linux desktop "${QT_VERSION}" gcc_64 -O "${SDK_DIR}"

# 2. Pre-extracted Qt Creator IDE (No .run login prompt on offline Ubuntu!)
run_aqt install-tool linux desktop tools_qtcreator -O "${SDK_DIR}"

# 3. CMake & Ninja Linux binaries
run_aqt install-tool linux desktop tools_cmake -O "${SDK_DIR}"
run_aqt install-tool linux desktop tools_ninja -O "${SDK_DIR}"

# Also download standalone Qt Creator .run installer if not already present
if [ ! -s "${OUT_DIR}/qt-creator-linux-x86_64.run" ]; then
    echo "Downloading standalone Qt Creator .run installer..."
    curl -L -C - --retry 10 --retry-delay 3 --speed-time 20 --speed-limit 10240 \
        "${MIRROR_PRIMARY}/official_releases/qtcreator/latest/qt-creator-opensource-linux-x86_64-20.0.2.run" \
        -o "${OUT_DIR}/qt-creator-linux-x86_64.run"
fi

# ------------------------------------------------------------------------------
# [4/5] Strict Verification Gate (Prevents incomplete/broken bundles)
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Running Verification Gate..."

QMAKE_BIN="${SDK_DIR}/${QT_VERSION}/gcc_64/bin/qmake"
QTC_BIN="${SDK_DIR}/Tools/QtCreator/bin/qtcreator"
DEB_COUNT=$(find "${DEB_DIR}" -maxdepth 1 -name "*.deb" | wc -l)

if [ ! -x "${QMAKE_BIN}" ]; then
    echo "[FATAL ERROR] ${QMAKE_BIN} is missing or not executable!"
    exit 1
fi

if [ ! -x "${QTC_BIN}" ]; then
    echo "[FATAL ERROR] ${QTC_BIN} is missing or not executable!"
    exit 1
fi

if [ "${DEB_COUNT}" -lt 20 ]; then
    echo "[FATAL ERROR] Only ${DEB_COUNT} .deb files found in ${DEB_DIR}. Expected 50+."
    exit 1
fi

echo "  [OK] Verified qmake binary       : ${QMAKE_BIN}"
echo "  [OK] Verified Qt Creator binary  : ${QTC_BIN}"
echo "  [OK] Verified Ubuntu .deb count  : ${DEB_COUNT} packages downloaded"

# Pack SDK with -p (preserve permissions and native Linux symlinks)
echo "Packing verified Qt_SDK into ${OUT_DIR}/qt_sdk_linux.tar.gz..."
tar -czpf "${OUT_DIR}/qt_sdk_linux.tar.gz" -C "${WORK_DIR}" Qt_SDK

# ------------------------------------------------------------------------------
# [5/5] Auto-Generate Offline Ubuntu Installer Script Inside Output Folder
# ------------------------------------------------------------------------------
echo ""
echo "[5/5] Generating install_on_offline_ubuntu.sh..."

cat << 'EOF' > "${OUT_DIR}/install_on_offline_ubuntu.sh"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] Installing Ubuntu C++ Compiler (g++) & System .deb Packages ==="
if compgen -G "${SCRIPT_DIR}/ubuntu_debs/*.deb" > /dev/null; then
    sudo dpkg -i --force-depends "${SCRIPT_DIR}/ubuntu_debs"/*.deb || true
    sudo dpkg --configure -a || true
else
    echo "ERROR: No .deb packages found in ${SCRIPT_DIR}/ubuntu_debs!"
    exit 1
fi

echo "=== [2/3] Extracting Qt SDK & Qt Creator to /opt/Qt ==="
sudo mkdir -p /opt/Qt
sudo tar -xzpf "${SCRIPT_DIR}/qt_sdk_linux.tar.gz" -C /opt/Qt --strip-components=1
sudo chmod -R a+rX /opt/Qt

echo "=== [3/3] Creating System Command & Desktop Shortcut for Qt Creator ==="
sudo ln -sf /opt/Qt/Tools/QtCreator/bin/qtcreator /usr/local/bin/qtcreator

cat << 'DESKTOP' | sudo tee /usr/share/applications/org.qt-project.qtcreator.desktop >/dev/null
[Desktop Entry]
Type=Application
Exec=/opt/Qt/Tools/QtCreator/bin/qtcreator %F
Name=Qt Creator (Offline)
GenericName=C++ IDE for developing Qt applications
Terminal=false
Categories=Development;IDE;Qt;
DESKTOP

echo "=================================================================="
echo " INSTALLATION COMPLETE!"
echo " 1. Run 'qtcreator' in terminal (or open it from the Apps menu)."
echo " 2. In Qt Creator -> Edit -> Preferences -> Kits:"
echo "    - Qt Versions -> Add: /opt/Qt/6.8.0/gcc_64/bin/qmake"
echo "    - Compilers   -> C++: /usr/bin/g++"
echo "    - CMake       -> Add: /opt/Qt/Tools/CMake/bin/cmake"
echo "=================================================================="
EOF

chmod +x "${OUT_DIR}/install_on_offline_ubuntu.sh"

ARCHIVE_SIZE=$(du -sh "${OUT_DIR}/qt_sdk_linux.tar.gz" | awk '{print $1}')
echo ""
echo "=================================================================="
echo " SUCCESS! Your Offline Bundle is 100% Verified and Ready:"
echo " Folder       : ${OUT_DIR}"
echo " SDK Archive  : qt_sdk_linux.tar.gz (${ARCHIVE_SIZE})"
echo " Deb Packages : ${DEB_COUNT} .deb files in ubuntu_debs/"
echo " Installer    : install_on_offline_ubuntu.sh"
echo "=================================================================="