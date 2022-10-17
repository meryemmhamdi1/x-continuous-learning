#!/usr/bin/env bash
for lang_order in "en_de_fr_hi_es_th" "th_es_hi_fr_de_en" "es_hi_en_de_th_fr" "fr_th_de_en_hi_es" "hi_en_es_th_fr_de" "de_fr_th_es_en_hi"
do
  for SIZE in 750 1500 3000 4500
  do
    sh scripts/train_trans_nlu_cll_er_kd.sh \
     "yes" \
     "BertBaseMultilingualCased" \
     "cll-er_kd" \
     0 \
     "en" \
     0 \
     $lang_order \
     "er" \
     "single_head" \
     "all" \
     "0_1_2_3_4_5_6" \
     "none" \
     "yes" \
     42 \
     "trans intent slot" \
     "MADX" \
     "reservoir" \
     "random" \
     $SIZE \
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
     "no"
  done
done
