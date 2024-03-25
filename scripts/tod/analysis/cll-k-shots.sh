#!/usr/bin/env bash

MODEL_OPT=${1:"vanilla"} # vanilla er kd-logits kd-rep ewc_online multi_head_in_all multi_head_out multi_head_in_enc8

if [ $MODEL_VER == "multi_head_in_all" ]; then
    MODEL="vanilla"
    MODEL_EXP="multi_head_in"
    COMPONENTS="embeddings_encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8._encoder.layer.9._encoder.layer.10._encoder.layer.11._pooler"
elif [ $MODEL_VER == "multi_head_in_enc8" ]; then
    MODEL="vanilla"
    MODEL_EXP="multi_head_in"
    COMPONENTS="encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8."
elif [ $MODEL_VER == "multi_head_out" ]; then
    MODEL="vanilla"
    MODEL_EXP="multi_head_out"
    COMPONENTS="all"
else
    MODEL=$MODEL_VER
    MODEL_EXP="single_head"
    COMPONENTS="all"
fi

for LANG in "en" "de" "fr" "hi" "es" "th"
do
  for K_SHOT_ORDER in 0 1
  do
    sh scripts/tod/train_trans_nlu_cll_er_kd.sh "yes" "BertBaseMultilingualCased" "cll-k-shots" $K_SHOT_ORDER $LANG 0 "en_de_fr_hi_es_th" $MODEL $MODEL_EXP $COMPONENTS

   
  done
done

