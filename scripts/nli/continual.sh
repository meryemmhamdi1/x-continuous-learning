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

# "zh_vi_ar_tr_bg_el_ur" "vi_ur_zh_ar_tr_bg_el" "ar_zh_el_ur_vi_tr_bg" "tr_ar_bg_el_ur_zh_vi" "bg_tr_vi_zh_el_ur_ar" "el_bg_ur_vi_zh_ar_tr" "ur_el_tr_bg_ar_vi_zh"
# for LANG_ORDER in "en_zh_vi_ar_tr_bg_el_ur" "ur_el_bg_tr_ar_vi_zh_en" "zh_vi_en_ur_el_ar_tr_bg" "vi_bg_zh_en_ur_tr_ar_el" "ar_tr_el_vi_en_ur_bg_zh" "tr_ur_ar_bg_zh_el_en_vi" "bg_en_tr_el_vi_zh_ur_ar" "el_ar_ur_zh_bg_en_vi_tr"
for LANG_ORDER in "en_vi_ar_tr_bg_el_ur" "vi_ur_en_ar_tr_bg_el" "ar_en_el_ur_vi_tr_bg" "tr_ar_bg_el_ur_en_vi" "bg_tr_vi_en_el_ur_ar" "el_bg_ur_vi_en_ar_tr" "ur_el_tr_bg_ar_vi_en" 
do
  for SEED in 42 #35 40
  do
    sh scripts/nli/train_trans_nli_cll_er_kd.sh \
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
      "zh vi ar tr bg el ur en" \
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
