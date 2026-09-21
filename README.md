# DSOM: Dual-Space Optimization via Test-Time Adaptation for Open-Set Activity Recognition

**Xiaohui Ye, Lei Zhang, Guangjie Chen, Chaoda Song, Shuoyuan Wang, Hao Wu, Aiguo Song, Senior Member, IEEE**

---

## Experimental Audit Trail

**The experiment logs and statistical tables behind every reported result are included in this repository under [`audit_trail/`](audit_trail/).**

All results in the manuscript were produced under a **unified five-seed protocol** — seeds **10, 42, 64, 100, 256** — using the same subject-level partitioning and validation-based hyperparameter selection throughout. The same protocol was applied to the main comparisons and to the ablation experiments.

| Experiment | `audit_trail/` contents |
| --- | --- |
| Main cross-person comparison | Per-seed logs for every method, plus the five-seed average and the metrics table |
| Ablation study (UCI-HAR / OPPORTUNITY / PAMAP2) | Per-seed ablation tables and the aggregate summary |
| Backbone experiments | CNN, ResNet, ViT, and DeepConvLSTM across the five seeds |
| Cross-category (open-set) evaluation | Per-seed open-set logs and score tables |
| Cross-person statistics | Metrics spreadsheets for the five-seed runs |

All reported means and aggregate values are computed directly from these seed-level runs.

---

DSOM is a test-time adaptation framework for sensor-based human activity recognition (HAR). It targets the practical setting in which test streams are non-i.i.d., arrive in small batches, and may contain activity categories never seen during training (open-set). Adaptation proceeds in two stages: aligning the feature space through backbone normalization, then refining prototype embeddings via energy-space optimization.



## Baselines

Methods compared against DSOM. The `--adaption` column gives the flag used to select each method in [`code/`](code/).

| Method | `--adaption` | Description |
| --- | --- | --- |
| ERM | `source` | Keeps model parameters fixed at test time as the canonical baseline for transfer learning. |
| T3A | `t3a` | Adapts the classifier by computing class prototypes from low-entropy samples. |
| OFTTA | `offta` | Refines predictions using Exponential Decay Batch Normalization and pseudo-prototypes without relying on gradient-based optimization. |
| PL | `pl` | Iteratively generates high-confidence pseudo-labels for online self-training and progressive adaptation. |
| TENT | `tent` | Minimizes prediction entropy by updating only BN affine parameters during inference. |
| LAME | `lame` | Enhances generalization via neighborhood consistency without parameter updates. |
| SHOT | `shot` | Adapts the model with entropy minimization, class balance, and structured pseudo-labeling. |
| TSD | `tsd` | Employs memory clustering mechanism, incorporating entropy and consistency filtering to mitigate noise from pseudo-label interference. |
| TAST | `tast` | Aligns prediction distributions using a trainable Batch Ensemble module and prototype classifiers. |
| TEA | `tea` | Transforms the classifier into an energy-based model and aligns via contrastive energy minimization. |
| NOTE | `note` | Combines Instance Aware-BN and Prediction-Balanced Reservoir Sampling for stable memory adaptation. |
| SoTTA | `sotta` | Adapts dynamically to non-stationary data streams by enforcing temporal consistency and minimizing prediction entropy. |
| SAR | `sar` | Introduces sharpness-aware entropy minimization with reliable-sample selection to improve adaptation stability and prevent model collapse during TTA. |
| EATA | `eata` | Extends entropy minimization with an anti-forgetting and sample-selective mechanism, filtering out non-informative samples to stabilize continual TTA. |
| RoTTA | `rotta` | Adopts a class-balanced memory sampling strategy and robust pseudo-labels for reliable continual test-time adaptation under non-stationary shifts. |
| CoTTA | `cotta` | Uses a mean-teacher framework with stochastic augmentation and restore operations to prevent error accumulation in long-running test-time adaptation. |
| **DSOM** | `dsom` | **Ours.** Two-stage adaptation: feature-space alignment via backbone normalization, followed by energy-space refinement of prototype embeddings. |

## Code

The implementation is in [`code/`](code/). It evaluates the methods above on UCI-HAR across target domains 0–4; see [`code/readme.md`](code/readme.md) for setup and execution details.

## Citation

If you use this work, please cite:

```bibtex
@article{ye2026dsom,
  title  = {DSOM: Dual-Space Optimization via Test-Time Adaptation for Open-Set Activity Recognition},
  author = {Ye, Xiaohui and Zhang, Lei and Chen, Guangjie and Song, Chaoda and Wang, Shuoyuan and Wu, Hao and Song, Aiguo},
  year   = {2026}
}
```

A [`CITATION.cff`](CITATION.cff) file is provided for citation managers.

## License

Text and figures in this repository are excerpted from the DSOM paper for documentation and reference purposes. All rights belong to the original authors. See [`LICENSE`](LICENSE).
