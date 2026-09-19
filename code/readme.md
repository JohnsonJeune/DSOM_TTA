## DSOM — Test-Time Adaptation (TTA) Benchmark

This project is based on the official implementation of [Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition](https://github.com/Claydon-Wang/OFTTA) (IMWUT/UbiComp 2024),
with 9 common TTA methods integrated on top of the original ones.

- Datasets: **UCI-HAR** **OPPORTUNITY** **PAMAP2**
- Backbone model: CNN
- Device: GPU via CUDA, with automatic CPU fallback

### Device Selection

`device.py` resolves `DEVICE` once at import time:

```python
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

Every tensor and module transfer in the codebase targets `DEVICE`, so the project runs on the first CUDA device when one is present and falls back to CPU otherwise. No code change is needed to switch between the two — only the environment changes.

### Directory Structure

```
adapt.py                     Evaluation entry point
device.py                    Shared DEVICE selection (CUDA when available, else CPU)
utils.py                     Dataset / model construction (uci + cnn only)
data_processing/             Dataset preprocessing + sliding window
models/                      Backbones (cnn / cnn_mix / adnn)
TTA/setup.py                 Method registration and dispatch
TTA/adapt_algorithm/         TTA method implementations
cfg/dataset/*.yaml           Dataset configuration (uci / oppo / pamap2)
cfg/algorithm/*.yaml         Per-method hyperparameter configuration
scripts/<dataset>/           Batch run scripts per method
adapt.sh                     Run every UCI-HAR method in one go
data/<dataset>/              Dataset files
ckpt/<dataset>/cnn/<domain>/ Pretrained source model weights
```

### Installation

```bash
conda create -y -n oftta python=3.9
conda activate oftta

# CUDA build (installs the matching CUDA runtime).
# For a CPU-only machine instead:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch
pip install -r requirements.txt
```

### Running

A single evaluation (OFTTA on UCI-HAR domain 0):

```bash
python adapt.py \
    --target_domain 0 \
    --dataset_cfg ./cfg/dataset/uci.yaml \
    --algorithm_cfg ./cfg/algorithm/oftta.yaml
```

Run all target domains for one method — each dataset uses its own domain set:

```bash
bash scripts/uci/adapt_oftta_uci.sh        # UCI-HAR,    domains 0 1 2 3 4
bash scripts/oppo/adapt_oftta_oppo.sh      # OPPORTUNITY, domains S1 S2 S3 S4
bash scripts/pamap2/adapt_oftta_pamap2.sh  # PAMAP2,     domains 1 2 3 4 5
```

Run every UCI-HAR method:

```bash
bash adapt.sh
```

If `python` is not on PATH or points to another environment, override it with the `PY` variable:

```bash
PY=/path/to/python bash adapt.sh
```

Results are written to `./logs/<dataset>/<method>/<domain>/<timestamp>/log.txt` by default, and contain `Source Accuracy` and `Adapt Accuracy`.

### Supported Methods

| Method | `--adaption` | Config file | Description |
|:--|:--|:--|:--|
| Source | `source` | `source.yaml` | Source-model baseline, no adaptation |
| NORM | `norm` | `norm.yaml` | Test-time batch-statistics normalization |
| TENT | `tent` | `tent.yaml` | Entropy minimization (ICLR 2021) |
| T3A | `t3a` | `t3a.yaml` | Test-time classifier adjustment (NeurIPS 2021) |
| TAST | `tast` | `tast.yaml` | Nearest-neighbor self-training (ICLR 2023) |
| TAST-BN | `tast_bn` | `tast_bn.yaml` | BN variant of TAST |
| OFTTA | `offta` | `offta.yaml` | This repository's original method (IMWUT 2024) |
| PL | `pl` | `pl.yaml` | Pseudo-labeling (ICML Workshop 2013) |
| SHOT | `shot` | `shot.yaml` | Source hypothesis transfer (ICML 2020) |
| SAR | `sar` | `sar.yaml` | Stable test-time adaptation (ICLR 2023) |
| **DSOM** | `dsom` | `dsom.yaml` | Dual-space optimization test-time adaptation |
| **EATA** | `eata` | `eata.yaml` | Efficient anti-forgetting TTA (ICML 2022) |
| **SoTTA** | `sotta` | `sotta.yaml` | Noise-robust TTA (NeurIPS 2023) |
| **CoTTA** | `cotta` | `cotta.yaml` | Continual TTA (CVPR 2022) |
| **TSD** | `tsd` | `tsd.yaml` | Test-time self-distillation (CVPR 2023) |
| **TEA** | `tea` | `tea.yaml` | Energy-model test-time adaptation |
| **NOTE** | `note` | `note.yaml` | Online entropy minimization (NeurIPS 2022) |
| **RoTTA** | `rotta` | `rotta.yaml` | Robust test-time adaptation |
| **LAME** | `lame` | `lame.yaml` | Laplacian label propagation (NeurIPS 2022) |
