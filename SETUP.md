# Setup

HFA-Mamba is a fork of [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
(AAAI 2025), which provides the `ultralytics/` source and the `selective_scan/`
CUDA extension. This repo ships only the HFA-Mamba **additions** (HFRM, SBL,
configs). Because HFRM and SBL are injected **at runtime** by `mbyolo_train.py`,
**no upstream file is modified** — integration is just "clone base + copy our files".

## Recommended: one command

```bash
git clone https://github.com/<your-org>/HFA-Mamba.git && cd HFA-Mamba
bash build_with_mambayolo.sh          # clones Mamba-YOLO and overlays HFA-Mamba
```

This produces `./HFA-Mamba-build/`, a complete tree (the full Mamba-YOLO source +
`selective_scan/` + HFA-Mamba's HFRM/SBL/configs). Then build the environment:

```bash
conda create -n hfa-mamba -y python=3.11 && conda activate hfa-mamba
pip3 install torch==2.3.0 torchvision torchaudio        # CUDA 12.1
pip install seaborn thop timm einops
cd HFA-Mamba-build
cd selective_scan && pip install . && cd ..             # selective-scan (from VMamba, via base)
pip install -v -e .                                     # editable install of the vendored fork
```

Train:

```bash
python mbyolo_train.py --task train \
  --data   ultralytics/cfg/datasets/afo.yaml \
  --config ultralytics/cfg/models/hfa-mamba/HFA-Mamba-B.yaml \
  --amp --imgsz 2160 --project ./output_dir/afo --name hfa_mamba_b
```

## Why ship an overlay + integrator instead of a pre-merged tree?

The upstream `ultralytics/` source (hundreds of files) and the `selective_scan/`
CUDA extension are large and already maintained upstream under AGPL-3.0. Shipping a
deterministic integrator keeps this repo small, always tracks the exact upstream
you pin, and avoids redistributing a stale copy. `build_with_mambayolo.sh` accepts a
repo URL and git ref so you can pin a commit:

```bash
bash build_with_mambayolo.sh https://github.com/HZAI-ZJNU/Mamba-YOLO.git <commit-sha>
```

## Manual overlay (no script)

```bash
git clone https://github.com/HZAI-ZJNU/Mamba-YOLO.git
rsync -a HFA-Mamba/ultralytics/ Mamba-YOLO/ultralytics/   # adds hfrm.py, sbl.py, cfgs
cp HFA-Mamba/mbyolo_train.py Mamba-YOLO/                   # runtime HFRM/SBL injection + val
cp -r HFA-Mamba/{docs,tests,asserts} Mamba-YOLO/
cd Mamba-YOLO   # then build env as above
```

HFRM and SBL depend only on PyTorch; the selective-scan kernels are needed only for
the Mamba backbone blocks (`VSSBlock` / `XSSBlock`).
