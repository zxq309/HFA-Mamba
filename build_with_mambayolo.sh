#!/usr/bin/env bash
# =============================================================================
# build_with_mambayolo.sh
#
# Produces a COMPLETE, runnable HFA-Mamba tree by integrating this overlay with
# the official Mamba-YOLO source. Because HFRM and SBL are injected at runtime by
# mbyolo_train.py, this script does NOT patch any upstream file -- it only clones
# the base and copies the HFA-Mamba additions on top.
#
# Usage:
#   bash build_with_mambayolo.sh [MAMBA_YOLO_REPO_URL] [GIT_REF]
#
# Env:
#   BUILD_DIR   output directory (default: ./HFA-Mamba-build)
# =============================================================================
set -euo pipefail

REPO_URL="${1:-https://github.com/HZAI-ZJNU/Mamba-YOLO.git}"
GIT_REF="${2:-main}"
OVERLAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-./HFA-Mamba-build}"

echo "==> [1/4] Cloning Mamba-YOLO base ($GIT_REF) -> $BUILD_DIR"
rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$GIT_REF" "$REPO_URL" "$BUILD_DIR"

echo "==> [2/4] Overlaying HFA-Mamba files (no upstream file is overwritten by logic)"
# --- new module files (additive) ---
install -D -m644 "$OVERLAY_DIR/ultralytics/nn/modules/hfrm.py" "$BUILD_DIR/ultralytics/nn/modules/hfrm.py"
install -D -m644 "$OVERLAY_DIR/ultralytics/utils/sbl.py"       "$BUILD_DIR/ultralytics/utils/sbl.py"
# --- configs (new files in existing dirs) ---
mkdir -p "$BUILD_DIR/ultralytics/cfg/models/hfa-mamba/ablation"
cp -r "$OVERLAY_DIR/ultralytics/cfg/models/hfa-mamba/." "$BUILD_DIR/ultralytics/cfg/models/hfa-mamba/"
cp     "$OVERLAY_DIR/ultralytics/cfg/datasets/"*.yaml    "$BUILD_DIR/ultralytics/cfg/datasets/"
# --- entry point (ours adds runtime HFRM/SBL injection + a val task) ---
cp "$OVERLAY_DIR/mbyolo_train.py" "$BUILD_DIR/mbyolo_train.py"
# --- docs / tests / readme / figures ---
cp -r "$OVERLAY_DIR/docs"    "$BUILD_DIR/docs"
mkdir -p "$BUILD_DIR/tests"; cp "$OVERLAY_DIR/tests/"*.py "$BUILD_DIR/tests/"
cp "$OVERLAY_DIR/README.md" "$BUILD_DIR/README.md"
cp "$OVERLAY_DIR/SETUP.md"  "$BUILD_DIR/SETUP.md"
cp -r "$OVERLAY_DIR/asserts" "$BUILD_DIR/asserts" 2>/dev/null || true

echo "==> [3/4] Sanity check: HFA-Mamba files in place"
ls "$BUILD_DIR/ultralytics/nn/modules/hfrm.py" \
   "$BUILD_DIR/ultralytics/utils/sbl.py" \
   "$BUILD_DIR/ultralytics/cfg/models/hfa-mamba/HFA-Mamba-B.yaml" >/dev/null && echo "    ok"

echo "==> [4/4] Done. Now build the environment:"
cat <<'NEXT'

    conda create -n hfa-mamba -y python=3.11 && conda activate hfa-mamba
    pip3 install torch==2.3.0 torchvision torchaudio
    pip install seaborn thop timm einops
    cd HFA-Mamba-build
    cd selective_scan && pip install . && cd ..      # CUDA selective-scan (from the base)
    pip install -v -e .                              # editable install of the vendored fork

    # train HFA-Mamba-B on AFO:
    python mbyolo_train.py --task train \
      --data   ultralytics/cfg/datasets/afo.yaml \
      --config ultralytics/cfg/models/hfa-mamba/HFA-Mamba-B.yaml \
      --amp --imgsz 2160 --project ./output_dir/afo --name hfa_mamba_b
NEXT