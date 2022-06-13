#!/usr/bin/env bash

for lang_order in "en_de_fr_hi_es_th" "th_es_hi_fr_de_en" "es_hi_en_de_th_fr" "fr_th_de_en_hi_es" "hi_en_es_th_fr_de" "de_fr_th_es_en_hi"
do
  for SEED in 42 35 40
  do
    echo "Multi-headed in OVER order "$lang_order
    sh scripts/train_trans_nlu_cll_er_kd.sh \
       "yes" \
       "BertBaseMultilingualCased" \
       "cll-er_kd" \
       0 \
       "en" \
       0 \
       $lang_order \
       "vanilla" \
       "multi_head_in" \
       "embeddings_encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8._encoder.layer.9._encoder.layer.10._encoder.layer.11._pooler" \
       "0_1_2_3_4_5_6" \
       "none" \
       "yes" \
       $SEED \
       "trans intent slot" \
       "MADX" \
       "ring" \
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
       "no"
  done
done