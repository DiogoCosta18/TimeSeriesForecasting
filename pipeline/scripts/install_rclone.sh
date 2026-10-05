#!/usr/bin/env bash
# Install the pinned rclone release (D20) into ~/.local/bin, verified against its published SHA256SUMS.
set -euo pipefail
V=v1.75.1
W=$HOME/scratch_rclone && rm -rf "$W" && mkdir -p "$W" && cd "$W"
curl -sSLO "https://github.com/rclone/rclone/releases/download/$V/rclone-$V-linux-amd64.zip"
curl -sSLO "https://github.com/rclone/rclone/releases/download/$V/SHA256SUMS"
grep " rclone-$V-linux-amd64.zip\$" SHA256SUMS | sha256sum -c -
python3 -c "import zipfile; zipfile.ZipFile('rclone-$V-linux-amd64.zip').extractall('.')"
mkdir -p "$HOME/.local/bin"
install -m 0755 "rclone-$V-linux-amd64/rclone" "$HOME/.local/bin/rclone"
"$HOME/.local/bin/rclone" version | head -3
