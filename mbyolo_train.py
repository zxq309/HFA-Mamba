"""HFA-Mamba train / val entry point (Mamba-YOLO CLI).

ZERO upstream edits: this script injects the two HFA-Mamba contributions into the
vendored Mamba-YOLO / Ultralytics fork *at runtime*:

  1. HFRM  — registered by name so the model YAML parser can resolve it. HFRM is
     channel-agnostic and preserves channels, so the parser's default branch
     handles it without any change to ``parse_model``.
  2. SBL   — installed by overriding the model's ``init_criterion`` to return
     ``SBLDetectionLoss`` (reads hyperparameters from sbl.yaml).

Because the vendored fork is installed editable (``pip install -v -e .``),
``from ultralytics import YOLO`` loads the *local* fork (with the Mamba kernels),
not the upstream pip package.

Examples
--------
Train HFA-Mamba-B on AFO:
    python mbyolo_train.py --task train \
        --data   ultralytics/cfg/datasets/afo.yaml \
        --config ultralytics/cfg/models/hfa-mamba/HFA-Mamba-B.yaml \
        --amp --imgsz 2160 --project ./output_dir/afo --name hfa_mamba_b

Ablate SBL off (Model B):
    python mbyolo_train.py --task train --no_sbl ... (same args)

Validate:
    python mbyolo_train.py --task val \
        --data ultralytics/cfg/datasets/afo.yaml \
        --weight ./output_dir/afo/hfa_mamba_b/weights/best.pt --imgsz 2160
"""

import argparse
import os
from types import MethodType

import yaml


def parse_opt():
    p = argparse.ArgumentParser(description="HFA-Mamba training / validation")
    p.add_argument("--task", type=str, default="train", choices=["train", "val"])
    p.add_argument("--data", type=str, required=True, help="dataset YAML")
    p.add_argument("--config", type=str, default="", help="model YAML (training)")
    p.add_argument("--weight", type=str, default="", help="checkpoint (val/resume)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=2160, help="HERIDAL 1536 / AFO 2160 / TinyPerson 1024")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--amp", action="store_true", help="mixed-precision training")
    p.add_argument("--project", type=str, default="./output_dir/hfa_mamba")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--seed", type=int, default=0)
    # SBL controls
    p.add_argument("--sbl_cfg", type=str,
                   default="ultralytics/cfg/models/hfa-mamba/sbl.yaml",
                   help="SBL hyperparameter YAML")
    p.add_argument("--no_sbl", action="store_true", help="disable SBL (use stock loss)")
    return p.parse_args()


def register_hfrm():
    """Make the name 'HFRM' resolvable inside parse_model (tasks.py globals)."""
    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules.hfrm import HFRM
    setattr(tasks, "HFRM", HFRM)


def load_sbl_cfg(path):
    if not os.path.isfile(path):
        return None
    cfg = (yaml.safe_load(open(path)) or {}).get("sbl", {})
    if not cfg.get("enabled", True):
        return None
    return {
        "alpha_r": float(cfg.get("alpha_r", 5e-4)),
        "beta_c": float(cfg.get("beta_c", 1.0)),
        "gamma_c": float(cfg.get("gamma_c", 800.0)),
    }


def install_sbl(model, sbl_kwargs):
    """Override init_criterion so the trainer uses SBLDetectionLoss."""
    from ultralytics.utils.sbl import SBLDetectionLoss

    def init_criterion(self):
        return SBLDetectionLoss(self, **sbl_kwargs)

    model.model.init_criterion = MethodType(init_criterion, model.model)
    model.model.criterion = None  # force re-creation via the override
    print(f"[HFA-Mamba] SBL enabled: {sbl_kwargs}")


def main(opt):
    register_hfrm()
    from ultralytics import YOLO

    if opt.task == "train":
        assert opt.config, "--config (model YAML) is required for training"
        model = YOLO(opt.config)
        if opt.weight:
            model = YOLO(opt.config).load(opt.weight)

        if not opt.no_sbl:
            sbl_kwargs = load_sbl_cfg(opt.sbl_cfg)
            if sbl_kwargs is not None:
                install_sbl(model, sbl_kwargs)
        else:
            print("[HFA-Mamba] SBL disabled (--no_sbl): using stock detection loss")

        model.train(
            data=opt.data, epochs=opt.epochs, batch=opt.batch_size, imgsz=opt.imgsz,
            device=opt.device, optimizer=opt.optimizer, lr0=opt.lr0,
            weight_decay=opt.weight_decay, amp=opt.amp,
            mosaic=1.0, mixup=0.1, degrees=0.0, translate=0.1, scale=0.5,
            seed=opt.seed, project=opt.project, name=opt.name,
        )
    else:  # val
        assert opt.weight, "--weight (checkpoint) is required for validation"
        model = YOLO(opt.weight)
        print(model.val(data=opt.data, imgsz=opt.imgsz, device=opt.device,
                        project=opt.project, name=opt.name))


if __name__ == "__main__":
    main(parse_opt())
