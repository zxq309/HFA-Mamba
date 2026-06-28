# Integration -- how HFRM and SBL plug into Mamba-YOLO

This repo is a fork of [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
(which vendors a fork of Ultralytics). HFA-Mamba adds exactly two novel pieces --
the **HFRM** module and the **Scale-Balanced Loss (SBL)** -- plus the model/dataset
configs.

## The default path: runtime injection (ZERO upstream edits)

`mbyolo_train.py` installs both contributions at runtime, so **no upstream file is
modified**. This is what `build_with_mambayolo.sh` relies on.

1. **HFRM registration.** HFRM is *channel-agnostic* (it infers `C` from the input
   and preserves it). The Ultralytics/Mamba-YOLO `parse_model` default branch
   already records `output_channels == input_channels` for an unrecognised module,
   so HFRM needs no change to `parse_model`. It only needs its **name** resolvable
   inside `tasks.py`. `mbyolo_train.py` does:

   ```python
   import ultralytics.nn.tasks as tasks
   from ultralytics.nn.modules.hfrm import HFRM
   tasks.HFRM = HFRM            # now the YAML token `HFRM` resolves
   ```

   After this, a YAML line `[-1, 1, HFRM, [256, 0.20]]` builds `HFRM(256, 0.20)`
   (the `256` is ignored; `0.20` is `radius_ratio`).

2. **SBL installation.** The model creates its loss lazily via
   `DetectionModel.init_criterion()`. `mbyolo_train.py` overrides it:

   ```python
   from types import MethodType
   from ultralytics.utils.sbl import SBLDetectionLoss
   model.model.init_criterion = MethodType(
       lambda self: SBLDetectionLoss(self, alpha_r=5e-4, beta_c=1.0, gamma_c=800.0),
       model.model)
   model.model.criterion = None   # force re-creation through the override
   ```

   Hyperparameters come from `ultralytics/cfg/models/hfa-mamba/sbl.yaml`. Disable
   with `--no_sbl` (or `enabled: false`) for the Model A / B ablations.

`SBLDetectionLoss` (in `ultralytics/utils/sbl.py`) subclasses `v8DetectionLoss` and
reuses its assigner, anchors, box decoding, BCE and `bbox_loss`. The only change is
two per-positive weights keyed on the image-normalized target-box area:
classification BCE x `lambda_cls` (boost small), regression weight x `lambda_reg`
(suppress tiny -- drives both CIoU and DFL, matching Eq. 17).

> Targets the Ultralytics 8.x loss API. If your vendored fork's `v8DetectionLoss`
> differs, re-sync the body of `SBLDetectionLoss.__call__` with it (the math is the
> two weights above), or use the optional baked-in path below.

## Optional: bake the edits into the fork

If you prefer the modules registered statically (e.g. to use the stock Ultralytics
CLI instead of `mbyolo_train.py`):

- `ultralytics/nn/modules/__init__.py`: `from .hfrm import HFRM` and add `"HFRM"`
  to `__all__`.
- `ultralytics/nn/tasks.py`: `from ultralytics.nn.modules import HFRM` (so it is in
  `parse_model`'s globals); no dispatch-set change is needed (HFRM is
  channel-agnostic and falls through the default branch).
- `ultralytics/nn/tasks.py` `DetectionModel.init_criterion`: return
  `SBLDetectionLoss(self, ...)` instead of `v8DetectionLoss(self)`.

## Reusing HFRM / SBL outside this fork

Both files depend only on PyTorch:

```python
from ultralytics.nn.modules.hfrm import HFRM
feat = HFRM(radius_ratio=0.20)(feat)            # (B,C,H,W) -> (B,C,H,W)

from ultralytics.utils.sbl import lambda_cls, lambda_reg
```

## Difference from SD Loss / NWD (paper Section 4.5)

- **SD Loss** reweights *within* the regression loss (IoU vs center-distance);
  classification untouched.
- **NWD** swaps the IoU metric for a Gaussian-Wasserstein distance.
- **SBL** reweights *across the two heads* (cls vs reg+DFL) with two independent,
  smooth, image-normalized functions -- decoupled, not forced to trade off.

On AFO, shared baseline: SBL 51.55 vs SD Loss 50.74 vs baseline 49.42
mAP@0.5:0.95; full framework 52.96 (SBL) vs 52.90 (NWD) vs 52.18 (SD Loss). Table 7.