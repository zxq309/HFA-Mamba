# Reproducing the Ablations

Every ablation in the paper is reproduced by selecting a model `--config` and/or
toggling SBL in `ultralytics/cfg/models/hfa-mamba/sbl.yaml`. All commands below use
AFO (`imgsz 2160`) unless noted; swap `--data` for HERIDAL / TinyPerson.

The four core models (Table 4) are the cross of {HFRM in YAML} x {SBL on/off}:

| Model | Config | SBL (`sbl.yaml`) |
|:---:|:---|:---:|
| A (baseline) | `ablation/HFA-Mamba-B_baseline.yaml` | `enabled: false` |
| B (+HFRM) | `HFA-Mamba-B.yaml` | `enabled: false` |
| C (+SBL) | `ablation/HFA-Mamba-B_baseline.yaml` | `enabled: true` |
| D (full) | `HFA-Mamba-B.yaml` | `enabled: true` |

```bash
# Model D (full HFA-Mamba-B) on AFO
python mbyolo_train.py --task train \
  --data   ultralytics/cfg/datasets/afo.yaml \
  --config ultralytics/cfg/models/hfa-mamba/HFA-Mamba-B.yaml \
  --amp --imgsz 2160 --project ./output_dir/afo --name model_D
```

## Table 5 -- SBL hyperparameter sensitivity

Sweep `alpha_r in {2.5e-4, 5e-4, 1e-3}`, `beta_c in {0.5, 1, 2}`,
`gamma_c in {400, 800, 1600}` by editing `hfa-mamba/sbl.yaml` and rerunning Model D.
Defaults are the middle of each range (`5e-4, 1.0, 800`).

## Table 6 -- HFRM mask radius `r/H'`

Edit the `0.20` argument in the three `HFRM` lines of `HFA-Mamba-B.yaml` to one of
`{0.10, 0.15, 0.20, 0.30, 0.50}` and rerun. (`0.20` is optimal on AFO.)

## Table 7 -- SBL vs SD Loss vs NWD

- **SBL**: Model D as above.
- **SD Loss / NWD**: implement the alternative in `utils/loss.py` (see
  `docs/integration.md` for the SBL hook point) and run with `sbl.yaml: enabled: false`.

## Table 8 -- HFRM integration level

| Placement | Config |
|:---|:---|
| P3 only | `ablation/HFA-Mamba-B_hfrm-p3.yaml` |
| P4 only | `ablation/HFA-Mamba-B_hfrm-p4.yaml` |
| P5 only | `ablation/HFA-Mamba-B_hfrm-p5.yaml` |
| P3 + P4 | `ablation/HFA-Mamba-B_hfrm-p3p4.yaml` |
| P3 + P4 + P5 (full) | `HFA-Mamba-B.yaml` |

All Table 8 runs keep SBL **on** (`enabled: true`); the reference is Model C
(baseline + SBL, no HFRM). Run each `--config` on all three datasets to fill the
table.