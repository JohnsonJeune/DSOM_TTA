#!/bin/bash
# UCI-HAR 全方法评测:每个脚本遍历 target_domain 0~4。
# 纯 CPU 运行;预训练权重在 ./ckpt/uci/cnn/<domain>/,数据在 ./data/uci/。
set -u

cd "$(dirname "$0")"

# ---- 原有方法 ----
bash scripts/uci/adapt_source_uci.sh
bash scripts/uci/adapt_norm_uci.sh
bash scripts/uci/adapt_t3a_uci.sh
bash scripts/uci/adapt_tent_uci.sh
bash scripts/uci/adapt_pl_uci.sh
bash scripts/uci/adapt_shot_uci.sh
bash scripts/uci/adapt_sar_uci.sh
bash scripts/uci/adapt_tast_uci.sh
bash scripts/uci/adapt_tast_bn_uci.sh
bash scripts/uci/adapt_oftta_uci.sh

# ---- 新增方法 ----
bash scripts/uci/adapt_dsom_uci.sh
bash scripts/uci/adapt_eata_uci.sh
bash scripts/uci/adapt_sotta_uci.sh
bash scripts/uci/adapt_cotta_uci.sh
bash scripts/uci/adapt_tsd_uci.sh
bash scripts/uci/adapt_tea_uci.sh
bash scripts/uci/adapt_note_uci.sh
bash scripts/uci/adapt_rotta_uci.sh
bash scripts/uci/adapt_lame_uci.sh
