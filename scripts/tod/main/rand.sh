#!/usr/bin/env bash

# Rand
for SEED in 42 40 35
do
  sh scripts/tod/train_trans_nlu_cll_er_kd.sh \
    "yes" \
    "BertBaseMultilingualCased" \
    "cll-er_kd" \
    0 \
    "en" \
    0 \
    "en_de_fr_hi_es_th" \
    "vanilla" \
    "single_head" \
    "all" \
    "0_1_2_3_4_5_6" \
    "none" \
    "yes" \
    $SEED \
    "trans_model gclassifier slot_classifier" \
    "MADX" \
    "reservoir" \
    "random" \
    6000 \
    16 \
    "no" \
    "no" \
    "0" \
    "no" \
    "en" \
    "de en es fr hi th" \
    "yes" \
    "yes" \
    10 \
    2 \
    "no" \
    "no" \
    "no" \
    "mtop" \
    "txt" \
    "fifo" \
    "ltn" \
    "main" \
    "yes"
done
