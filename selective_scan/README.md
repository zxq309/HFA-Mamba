# selective_scan

The selective-scan CUDA extension (from [VMamba](https://github.com/MzeroMiko/VMamba),
as used by [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)) is **not bundled**
here. Obtain it from the Mamba-YOLO base, then build:

```bash
# copy the upstream selective_scan/ contents into this folder, then:
cd selective_scan && pip install . && cd ..
```

This extension is required only for the Mamba backbone blocks (`XSSBlock` /
`VSSBlock`). The HFRM module and the Scale-Balanced Loss are pure PyTorch and do
not need it.
