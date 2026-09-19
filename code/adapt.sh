#!/bin/bash
# UCI-HAR evaluation of all methods: each script iterates over target_domain 0~4.
# Runs on pure CPU; pretrained weights are in ./ckpt/uci/cnn/<domain>/, data in ./data/uci/.
set -u

cd "$(dirname "$0")"

# ---- existing methods ----
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

# ---- newly added methods ----
bash scripts/uci/adapt_dsom_uci.sh
bash scripts/uci/adapt_eata_uci.sh
bash scripts/uci/adapt_sotta_uci.sh
bash scripts/uci/adapt_cotta_uci.sh
bash scripts/uci/adapt_tsd_uci.sh
bash scripts/uci/adapt_tea_uci.sh
bash scripts/uci/adapt_note_uci.sh
bash scripts/uci/adapt_rotta_uci.sh
bash scripts/uci/adapt_lame_uci.sh
