#!/usr/bin/env bash

# Main script for executing different baselines which involve (or not) spaced repetition
DATASET_NAME=${1:-"mtop"} # multiatis
USE_LEITNER_QUEUE=${2:-"no"}
DEMOTE_TO_FIRST_DECK=${3:-"no"}
CONTINUAL_MULTI=${4:-"no"} # DOESN'T MATTER HERE
FIFO_RAND_MODE=${5:-"fifo"} # rand
LTN_SCHEDULER_TYPE=${6:-"ltn"} # rbf
ER_LQ_SCHEDULER_TYPE=${7:-"main"}

# Baseline: "mtop" "no"

# Leitner Queues:
## Demote First: "mtop" "yes" "yes" "no"
## Demote Previous: "mtop" "yes" "no" "no"

if [ $DATASET_NAME == "mtop" ]; then
    source scripts/tod/hyperparameters/mtop/config.sh
else # multiatis
    source scripts/tod/hyperparameters/multiatis/config.sh
fi

echo $DATASET_NAME $DATASET_FORMAT $LANGUAGES 

for SEED in 42 #35 40
do
  sh scripts/nli/train_trans_nlu_cll_er_kd.sh \
    "yes" \
    "BertBaseMultilingualCased" \
    "multi" \
    0 \
    "en" \
    0 \
    ${LANG_ORDERS_LST[1]} \
    "vanilla" \
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
    "no" \
    "en" \
    $LANGUAGES \
    "yes" \
    "yes" \
    10 \
    2 \
    $USE_LEITNER_QUEUE \
    $DEMOTE_TO_FIRST_DECK \
    $CONTINUAL_MULTI \
    $DATASET_NAME \ 
    $DATASET_FORMAT \
    $FIFO_RAND_MODE \
    $LTN_SCHEDULER_TYPE \
    $ER_LQ_SCHEDULER_TYPE
done

