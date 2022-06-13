#!/usr/bin/env bash

for lang in "en" "de" "fr" "hi" "es" "th"
do
  for SEED in 42 40 35
  do
    sh scripts/train_trans_nlu_cll_er_kd.sh \
        "yes" \
        "BertBaseMultilingualCased" \
        "cll-er_kd" \
        0 \
        "en" \
        0 \
        $lang \
        "vanilla" \
        "single_head" \
        "all" \
        "0_1_2_3_4_5_6" \
        "none" \
        "yes" \
        $SEED \
        "trans intent slot" \
        "MADX" \
        "reservoir" \
        "random" \
        6000 \
        16 \
        "no" \
        "no" \
        "0" \
        "yes" \
        $lang \
        $lang \
        "yes" \
        "yes" \
        10 \
        2 \
        "no" \
        "no"
  done
done

