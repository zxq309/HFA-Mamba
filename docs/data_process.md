# Data Preparation

HFA-Mamba is evaluated on three public aerial person-detection benchmarks. Convert
each to YOLO format and point the dataset YAMLs in
`ultralytics/cfg/datasets/` at them.

## Target layout (YOLO format)

```
datasets/
|-- <DATASET>/
    |-- images/{train,val,test}/   0001.jpg ...
    |-- labels/{train,val,test}/   0001.txt ...   # per object: cls cx cy w h (normalized 0-1)
```

The dataset YAMLs use `path: ./datasets/<DATASET>` with `train/val/test` subpaths;
adjust `path` to wherever you store the data.

| Dataset | `imgsz` | Notes | Source |
|:---|:---:|:---|:---|
| **HERIDAL** | 1536 | >68,750 patches; small, camouflaged persons | Bozic-Stulic et al., IJCV 2019 |
| **AFO** | 2160 | maritime SaR; 99% of objects < 1% image area; crowded | Gasienica-Jozkowy et al., ICAE 2021 |
| **TinyPerson** | 1024 | wide-view; diverse poses, high density; `sea/earth_person` -> `person` | Hong et al., GRSL 2022 |

## Evaluation

Tools follow the COCO protocol, reporting Precision, Recall, mAP@0.5,
mAP@0.5:0.95, and **AP_S** (area < 32x32 px). For TinyPerson you may additionally
evaluate with the official `MR^-2` miss-rate protocol from TinyBenchmark for
comparison with prior work.