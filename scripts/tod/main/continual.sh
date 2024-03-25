#!/usr/bin/env bash

MODEL_VER=${1:-"vanilla"} # "joint" "multi" "vanilla" "ewc_online" "er" "kd-logits" "kd-rep" "multi_head_in_all" "multi_head_out" "multi_head_in_enc8"
ER_SIZE=${2:-10105} # or 750 1500 3000 4500 6000 for ablation of ER size

if [ $MODEL_VER == "joint" ]; then
    SETUP_OPT="multi-incr-cll"
    MODEL="vanilla"
    MODEL_EXP="single_head"
    COMPONENTS="all"
elif [ $MODEL_VER == "multi" ]; then
    SETUP_OPT="multi"
    MODEL="vanilla"
    MODEL_EXP="single_head"
    COMPONENTS="all"
elif [ $MODEL_VER == "multi_head_in_all" ]; then
    SETUP_OPT="cll-er_kd"
    MODEL="vanilla"
    MODEL_EXP="multi_head_in"
    COMPONENTS="embeddings_encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8._encoder.layer.9._encoder.layer.10._encoder.layer.11._pooler"
elif [ $MODEL_VER == "multi_head_in_enc8" ]; then
    SETUP_OPT="cll-er_kd"
    MODEL="vanilla"
    MODEL_EXP="multi_head_in"
    COMPONENTS="encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8."
elif [ $MODEL_VER == "multi_head_out" ]; then
    SETUP_OPT="cll-er_kd"
    MODEL="vanilla"
    MODEL_EXP="multi_head_out"
    COMPONENTS="all"
else    
    SETUP_OPT="cll-er_kd"
    MODEL=$MODEL_VER
    MODEL_EXP="single_head"
    COMPONENTS="all"
fi

# EWC-Online
# for LANG_ORDER in "en_de_fr_hi_es_th" "th_es_hi_fr_de_en" "es_hi_en_de_th_fr" "fr_th_de_en_hi_es" "hi_en_es_th_fr_de" "de_fr_th_es_en_hi"
for LANG_ORDER in "en_de_hi_th" "th_hi_de_en" "hi_th_en_de" "de_en_th_hi"
do
  for SEED in 42 35 40
  do
    sh scripts/tod/train_trans_nlu_cll_er_kd.sh \
       "yes" \
       "BertBaseMultilingualCased" \
       $SETUP_OPT \
       0 \
       "en" \
       0 \
       $LANG_ORDER \
       $MODEL \
       $MODEL_EXP \
       $COMPONENTS \
       "0_1_2_3_4_5_6" \
       "none" \
       "yes" \
       $SEED \
       "trans_model gclassifier slot_classifier" \
       "MADX" \
       "reservoir" \
       "random" \
       $ER_SIZE
  done
done
