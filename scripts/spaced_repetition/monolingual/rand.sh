#!/usr/bin/env bash

for lang in "en" #"de" "fr" "hi" "es" "th"
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
      42 \
      "trans intent slot" \
      "MADX" \
      "ring" \
      "random" \
      6000 10 "no" \
      "no" \
      "0" \
      "yes" \
      $lang
done
