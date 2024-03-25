#!/usr/bin/env bash

# Main script for executing different baselines which involve (or not) spaced repetition
DATASET_NAME=${1:-"mtop"} # multiatis
SETUP=${2:-"cll-er_kd"} 
MODEL=${3:-"vanilla"} 
USE_LEITNER_QUEUE=${4:-"no"}
DEMOTE_TO_FIRST_DECK=${5:-"no"}
FIFO_RAND_MODE=${6:-"fifo"} # or rand
LTN_SCHEDULER_TYPE=${7:-"ltn"} # or rbf
ER_LQ_SCHEDULER_TYPE=${8:-"main"}

# Baseline: "mtop" "cll-er_kd" "vanilla" "no"

# Leitner Queues:
## Demote First: "mtop" "cll-er_kd" "vanilla" "yes" "yes"
## Demote Previous: "mtop" "cll-er_kd" "vanilla" "yes" "no"

if [ $DATASET_NAME == "mtop" ]; then
    source scripts/tod/hyperparameters/mtop/config.sh
else # multiatis
    source scripts/tod/hyperparameters/multiatis/config.sh
fi

echo $DATASET_NAME $DATASET_FORMAT $LANGUAGES 

for LANG in "${LANG_LST[@]}"
do
  for SEED in 42 #35 40
  do
    sh scripts/tod/train_trans_nlu_cll_er_kd.sh \
      "yes" \
      "BertBaseMultilingualCased" \
      $SETUP \
      0 \
      $LANG \
      0 \
      $LANG \
      $MODEL \
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
      "yes" \
      $LANG \
      $LANG \
      "yes" \
      "yes" \
      10 \
      2 \
      $USE_LEITNER_QUEUE \
      $DEMOTE_TO_FIRST_DECK \
      "no" \
      $DATASET_NAME \ 
      $DATASET_FORMAT \
      $FIFO_RAND_MODE \
      $LTN_SCHEDULER_TYPE \
      $ER_LQ_SCHEDULER_TYPE
  done
done
