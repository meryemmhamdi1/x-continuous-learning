#!/usr/bin/env bash

# Main script for executing different baselines which involve (or not) spaced repetition
SETUP=${1:-"cll-er_kd"} 
MODEL=${2:-"vanilla"} 
USE_LEITNER_QUEUE=${3:-"no"}
DEMOTE_TO_FIRST_DECK=${4:-"no"}
CONTINUAL_MULTI=${5:-"no"}
FIFO_RAND_MODE=${6:-"fifo"} # or rand
LTN_SCHEDULER_TYPE=${7:-"ltn"} # or rbf
ER_LQ_SCHEDULER_TYPE=${8:-"main"}

# Baselines:
## Naive Seq FT: "cll-er_kd" "vanilla" "no"
## ER: "cll-er_kd" "er" "no"
## Inc Joint: "multi-incr-cll" "vanilla" "no"

# Leitner Queues:
## Monolingual:
### Demote First: "cll-er_kd" "vanilla" "yes" "yes" "yes"
### Demote Previous: "cll-er_kd" "vanilla" "yes" "no" "yes"
## Cross-lingual:
### Demote First: "cll-er_kd" "vanilla" "yes" "yes" "no"
### Demote Previous: "cll-er_kd" "vanilla" "yes" "no" "no"

for LANG_ORDER in "en_ru_ar_id_fi_bn_te_sw" "sw_te_bn_fi_id_ar_ru_en" "ru_ar_en_sw_te_id_fi_bn" "ar_bn_ru_en_sw_fi_id_te" "id_fi_te_ar_en_sw_bn_ru" "fi_sw_id_bn_ru_te_en_ar" "bn_en_fi_te_ar_ru_sw_id" "te_id_sw_ru_bn_en_ar_fi"
do
  for SEED in 42 #35 40
  do
    sh scripts/qa/train_trans_qa_cll_er_kd.sh \
      "no" \
      "BertBaseMultilingualCased" \
      $SETUP \
      0 \
      "en" \
      0 \
      $LANG_ORDER \
      $MODEL \
      "single_head" \
      "all" \
      "0_1_2_3_4_5_6" \
      "none" \
      "yes" \
      $SEED \
      "trans gclassifier" \
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
      "en ru ar id fi bn te sw" \
      "yes" \
      "yes" \
      10 \
      2 \
      $USE_LEITNER_QUEUE \
      $DEMOTE_TO_FIRST_DECK \
      $CONTINUAL_MULTI \
      $FIFO_RAND_MODE \
      $LTN_SCHEDULER_TYPE \
      $ER_LQ_SCHEDULER_TYPE
  done
done
