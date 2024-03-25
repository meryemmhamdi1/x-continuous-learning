#!/usr/bin/env bash

# Main script for executing different baselines which involve (or not) spaced repetition
USE_LEITNER_QUEUE=${1:-"no"}
DEMOTE_TO_FIRST_DECK=${2:-"no"}
CONTINUAL_MULTI=${3:-"no"}
FIFO_RAND_MODE=${4:-"fifo"} # or rand
LTN_SCHEDULER_TYPE=${5:-"ltn"} # or rbf
ER_LQ_SCHEDULER_TYPE=${6:-"main"}

# Baseline: "multi" "vanilla" "no"

# Leitner Queues:
## Demote First: "yes" "yes" "no"
## Demote Previous: "yes" "no" "no"


for SEED in 42 #35 40
do
  sh scripts/nli/train_trans_nli_cll_er_kd.sh \
    "no" \
    "BertBaseMultilingualCased" \
    "multi" \
    0 \
    "en" \
    0 \
    "zh_vi_ar_tr_bg_el_ur" \
    "vanilla" \
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

