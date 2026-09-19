## DSOM — 测试时自适应(TTA)评测工程

本工程基于 [Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition](https://github.com/Claydon-Wang/OFTTA)(IMWUT/UbiComp 2024)官方实现,
并在原有方法基础上集成了 9 个常见的 TTA 方法。

- 数据集:**UCI-HAR** **OPPORTUNITY** **PAMAP2**
- 后端模型:CNN

### 目录结构

```
adapt.py                评测入口(CPU 版)
cpu_env.py              CPU 兜底垫片(把 torch 的 CUDA 接口降级为 no-op)
utils.py                数据集 / 模型构建(只保留 uci + cnn)
data_processing/        uci 数据预处理 + 滑窗
models/                 骨干网络(cnn / cnn_mix / adnn)
TTA/setup.py            方法注册与分发
TTA/adapt_algorithm/    各 TTA 方法实现
cfg/dataset/uci.yaml    数据集配置
cfg/algorithm/*.yaml    各方法的超参配置
scripts/uci/            各方法的批量运行脚本(遍历 domain 0~4)
adapt.sh                一键跑完全部方法
data/uci/               UCI-HAR 数据
ckpt/uci/cnn/<domain>/  预训练源模型权重
```

### 安装

```bash
conda create -y -n oftta python=3.9
conda activate oftta

# 务必用 CPU index-url 装 torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 运行

单次评测(以 OFTTA 在 domain 0 上为例):

```bash
python adapt.py \
    --target_domain 0 \
    --dataset_cfg ./cfg/dataset/uci.yaml \
    --algorithm_cfg ./cfg/algorithm/oftta.yaml
```

批量跑完某个方法的全部 domain:

```bash
bash scripts/uci/adapt_oftta_uci.sh
```

一键跑完全部方法:

```bash
bash adapt.sh
```

如果 `python` 不在 PATH 或指向了别的环境,用 `PY` 变量覆盖:

```bash
PY=/path/to/python bash adapt.sh
```

结果默认写到 `./logs/<dataset>/<method>/<domain>/<时间戳>/log.txt`,含 `Source Accuracy` 与 `Adapt Accuracy`。

### 支持的方法

| 方法 | `--adaption` | 配置文件 | 说明 |
|:--|:--|:--|:--|
| Source | `source` | `source.yaml` | 源模型基线,不做自适应 |
| NORM | `norm` | `norm.yaml` | 测试时批统计归一化 |
| TENT | `tent` | `tent.yaml` | 熵最小化(ICLR 2021) |
| T3A | `t3a` | `t3a.yaml` | 测试时分类器调整(NeurIPS 2021) |
| TAST | `tast` | `tast.yaml` | 最近邻自训练(ICLR 2023) |
| TAST-BN | `tast_bn` | `tast_bn.yaml` | TAST 的 BN 变体 |
| OFTTA | `offta` | `offta.yaml` | 本仓库方法(IMWUT 2024) |
| PL | `pl` | `pl.yaml` | 伪标签(ICML Workshop 2013) |
| SHOT | `shot` | `shot.yaml` | 源假设迁移(ICML 2020) |
| SAR | `sar` | `sar.yaml` | 稳定测试时自适应(ICLR 2023) |
| **DSOM** | `dsom` | `dsom.yaml` | 双空间优化测试时自适应 |
| **EATA** | `eata` | `eata.yaml` | 高效抗遗忘 TTA(ICML 2022) |
| **SoTTA** | `sotta` | `sotta.yaml` | 噪声鲁棒 TTA(NeurIPS 2023) |
| **CoTTA** | `cotta` | `cotta.yaml` | 持续 TTA(CVPR 2022) |
| **TSD** | `tsd` | `tsd.yaml` | 测试时自蒸馏(CVPR 2023) |
| **TEA** | `tea` | `tea.yaml` | 能量模型测试时自适应 |
| **NOTE** | `note` | `note.yaml` | 在线熵最小化(NeurIPS 2022) |
| **RoTTA** | `rotta` | `rotta.yaml` | 鲁棒测试时自适应 |
| **LAME** | `lame` | `lame.yaml` | 拉普拉斯标签传播(NeurIPS 2022) |



