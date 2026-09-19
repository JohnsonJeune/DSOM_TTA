## OFTTA — CPU-only Test-Time Adaptation (TTA) Benchmark

This project is based on the official implementation of [Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition](https://github.com/Claydon-Wang/OFTTA) (IMWUT/UbiComp 2024),
adapted to run **on CPU only**, with 9 common TTA methods integrated on top of the original ones.

- Dataset: **UCI-HAR** (5 domains, leave-one-domain-out cross-validation)
- Device: **CPU only**, no GPU / CUDA required
- Backbone model: CNN

### Directory Structure

```
adapt.py                Evaluation entry point (CPU version)
cpu_env.py              CPU fallback shim (downgrades torch CUDA APIs to no-ops)
utils.py                Dataset / model construction (uci + cnn only)
data_processing/        UCI data preprocessing + sliding window
models/                 Backbones (cnn / cnn_mix / adnn)
TTA/setup.py            Method registration and dispatch
TTA/adapt_algorithm/    TTA method implementations
cfg/dataset/uci.yaml    Dataset configuration
cfg/algorithm/*.yaml    Per-method hyperparameter configuration
scripts/uci/            Batch run scripts per method (iterates domains 0-4)
adapt.sh                Run every method in one go
data/uci/               UCI-HAR data
ckpt/uci/cnn/<domain>/  Pretrained source model weights
```

### Installation

```bash
conda create -y -n oftta python=3.9
conda activate oftta

# Make sure to install torch from the CPU index-url
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Running

A single evaluation (OFTTA on domain 0):

```bash
python adapt.py \
    --target_domain 0 \
    --dataset_cfg ./cfg/dataset/uci.yaml \
    --algorithm_cfg ./cfg/algorithm/oftta.yaml
```

Run all domains for one method:

```bash
bash scripts/uci/adapt_oftta_uci.sh
```

Run every method:

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
| **OFTTA** | `offta` | `offta.yaml` | This repository's original method (IMWUT 2024) |
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

Bold entries are methods added or reworked here.

### Notes on CPU Execution

The main path (`adapt.py` / `utils.py` / `models/` / `data_processing/` / all methods above) is native CPU code and runs as-is.

`cpu_env.py` is a fallback shim: it downgrades interfaces such as `torch.Tensor.cuda()` to return in place, preventing any legacy file that has not yet been ported from raising an error on a GPU-less machine. It is imported at the top of `adapt.py`, ahead of any TTA algorithm module.

### Citation

```bibtex
@article{wang2024optimization,
  title={Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition},
  author={Wang, Shuoyuan and Wang, Hangwei and Wang, Jindong and Xie, Xin and others},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year={2024}
}
```
