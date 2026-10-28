#!/bin/bash
# Run this on the OFFLINE Ubuntu 24.04 PC

BUNDLE_DIR="$HOME/qt5_offline_bundle"

if [ ! -d "$BUNDLE_DIR/debs" ]; then
    echo "ERROR: Could not find $BUNDLE_DIR/debs."
    echo "Please extract the transferred bundle directly into your Home folder."
    exit 1
fi

echo "1. Backing up system repository lists (Safeguard)..."
sudo cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak 2>/dev/null || true
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true

echo "2. Temporarily disabling online repositories to prevent apt from hanging..."
sudo sed -i 's/^Types: deb/#Types: deb/g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
sudo sed -i 's/^deb/#deb/g' /etc/apt/sources.list 2>/dev/null || true

echo "3. Registering the local bundle as a trusted offline repository..."
echo "deb [trusted=yes] file:$BUNDLE_DIR/debs ./" | sudo tee /etc/apt/sources.list.d/offline-qt5.list

echo "4. Refreshing apt with local packages..."
sudo apt-get update

echo "============================================================"
echo "5. [SAFEGUARD] Running a simulation (Dry-Run)..."
echo "============================================================"
sudo apt-get install --simulate build-essential cmake qtbase5-dev qt5-qmake qtbase5-dev-tools qttools5-dev qtcreator

echo " "
echo ">>> Review the output above."
echo ">>> Verify that it says '0 to remove'. If safe, proceed."
read -p "Does the simulation look safe? Proceed with actual install? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Installation aborted to protect the OS."
    exit 1
fi

echo "6. Proceeding with safe installation..."
sudo apt-get install -y build-essential cmake qtbase5-dev qt5-qmake qtbase5-dev-tools qttools5-dev qtcreator

echo "============================================================"
echo "Installation complete!"
