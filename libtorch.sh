#!/usr/bin/env bash

# Generated with AI from PS script for Linux install, use with caution.
# Download and extract LibTorch into ./external/libtorch (Linux).
# Usage:
#   LIBTORCH_URL="https://..." ./scripts/libtorch.sh
set -euo pipefail

# Resolve script directory (works when invoked from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
TORCH_DEST="$EXTERNAL_DIR/libtorch"
TORCH_ZIP="$EXTERNAL_DIR/libtorch.zip"

# Default URL (override with LIBTORCH_URL env var)
: "${LIBTORCH_URL:=https://download.pytorch.org/libtorch/nightly/cu132/libtorch-shared-with-deps-latest.zip}"

mkdir -p "$EXTERNAL_DIR"

if [ -d "$TORCH_DEST" ]; then
  echo "LibTorch already exists at $TORCH_DEST - skipping download."
  exit 0
fi

echo "Downloading LibTorch from: $LIBTORCH_URL"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$TORCH_ZIP" "$LIBTORCH_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TORCH_ZIP" "$LIBTORCH_URL"
else
  echo "Error: neither curl nor wget found. Install one and retry." >&2
  exit 1
fi

echo "Extracting LibTorch to $EXTERNAL_DIR ..."
# Try unzip first (most libtorch packages are zip). Fallback to tar if needed.
if command -v unzip >/dev/null 2>&1; then
  unzip -q "$TORCH_ZIP" -d "$EXTERNAL_DIR"
else
  # If it's a tarball
  if tar -tf "$TORCH_ZIP" >/dev/null 2>&1; then
    tar -xf "$TORCH_ZIP" -C "$EXTERNAL_DIR"
  else
    echo "Warning: unzip not available and file is not a tar archive. Attempting to use bsdtar..."
    if command -v bsdtar >/dev/null 2>&1; then
      bsdtar -xf "$TORCH_ZIP" -C "$EXTERNAL_DIR"
    else
      echo "Error: cannot extract archive (no unzip/tar/bsdtar)." >&2
      exit 1
    fi
  fi
fi

# Many libtorch zips extract to a top-level 'libtorch' directory.
# If extraction created a different directory, normalize it to external/libtorch.
if [ -d "$TORCH_DEST" ]; then
  echo "LibTorch extracted to $TORCH_DEST"
else
  # find first directory inside externalDir and move/rename it
  first_dir="$(find "$EXTERNAL_DIR" -maxdepth 1 -type d ! -path "$EXTERNAL_DIR" | head -n 1 || true)"
  if [ -n "$first_dir" ]; then
    mv "$first_dir" "$TORCH_DEST"
    echo "Moved $first_dir -> $TORCH_DEST"
  else
    echo "Error: expected extracted libtorch directory not found." >&2
    exit 1
  fi
fi

# Cleanup archive
rm -f "$TORCH_ZIP"

# Set permissive read/execute permissions for directories and read for files
chmod -R a+rX "$TORCH_DEST" || true

echo "LibTorch setup complete: $TORCH_DEST"