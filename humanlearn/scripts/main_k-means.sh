#!/usr/bin/env bash

# sh tests/scripts/main.sh $TASK_NAME $LANG_ORDER "cont-mono" "no" $LTN_MODEL "fifo" "yes" 5 "epoch" "er-main" "random"
TASK=${1:-"mtop"} # mtop, xnli, tydiqa, panx
LANG_ORDER=${2:-0} # 0 1 2 3 4 5 6 7
MODE=${3:-"cont-mono"} # "mono", "cont-mono", "cont-multi", "multi", "multi-incr-cll"
USE_LEITNER=${4:-"yes"} # no
LTN_MODEL=${5:-"ltn"} # rbf
RANDOM_BASELINE=${6:-"fifo"} # "rand" "rand-prop"
USE_ER=${7:-"no"} # yes
NUM_DECKS=${8:-5} # 3, 4, 5, 6, 7
BATCH_EPOCH=${9:-"epoch"} # batch
ER_LQ_SCHEDULER_TYPE=${10:-"er-both"} # er-main er-only
ER_STRATEGY=${11:-"hard"} # "easy", "extreme", "balanced", "random", "equal-lang", "per_movement" (NOT YET), "exponential" (NOT YET)
UPDATE_EVERYTHING=${12:-"everything"}

## EXAMPLES
# ER LEITNER BOTH HARD: "xnli" 0 "cont-multi" "yes" "ltn" "fifo" "yes" 5 "yes" "er-both" "hard"
# ER LEITNER BOTH EASY: "xnli" 0 "cont-multi" "yes" "ltn" "fifo" "yes" 5 "yes" "er-both" "easy"
# ER LEITNER BOTH EXTREME: "xnli" 0 "cont-multi" "yes" "ltn" "fifo" "yes" 5 "yes" "er-both" "extreme"
# ER LEITNER BOTH BALANCED: "xnli" 0 "cont-multi" "yes" "ltn" "fifo" "yes" 5 "yes" "er-both" "balanced"
# ER LEITNER BOTH RANDOM: "xnli" 0 "cont-multi" "yes" "ltn" "rand" "yes" 5 "yes" "er-both" "easy"

# TODO ER RANDOM WITH LEITNER QUEUES
# TODO WRITE ER WITHOUT LEITNER QUEUES 

# Dataset Parameters

DATASET_PARAMS=""
if [ $TASK == "mtop" ]; then
   DATASET_PARAMS+="--task_name tod --data_name mtop --data_format txt --use_slots --use_crf"
   ORDER_LST=("en_de_hi_th" "th_hi_de_en" "hi_th_en_de" "de_en_th_hi" "en" "de" "hi" "th") 
   MAX_MEM_SZ=10105
elif [ $TASK == "xnli" ]; then
   DATASET_PARAMS+="--task_name nli --data_name xnli --data_format tsv"
   ORDER_LST=("en_vi_ar_tr" "tr_ar_vi_en" "ar_tr_en_vi" "vi_en_tr_ar" "vi" "en" "tr" "ar")
   MAX_MEM_SZ=1000
elif [ $TASK == "tydiqa" ]; then
   DATASET_PARAMS+="--task_name qa --data_name tydiqa --data_format json"
   ORDER_LST=("ru_id_te_sw" "sw_te_id_ru" "te_sw_ru_id" "id_ru_sw_te" "id" "ru" "sw" "te")
   MAX_MEM_SZ=500
elif [ $TASK == "panx" ]; then
   DATASET_PARAMS+="--task_name ner --data_name panx --data_format txt"
   ORDER_LST=("ru_id_te_sw" "sw_te_id_ru" "te_sw_ru_id" "id_ru_sw_te" "id" "ru" "sw" "te")
   MAX_MEM_SZ=1000
fi
# Extra Parameters

EXTRA_PARAMS=""
if [ $USE_LEITNER == "yes" ]; then
   EXTRA_PARAMS+="--use_leitner "
fi

if [ $USE_ER == "yes" ]; then
   EXTRA_PARAMS+="--use_er --er_lq_scheduler_type $ER_LQ_SCHEDULER_TYPE --er_strategy $ER_STRATEGY "
fi

if [ $MODE == "multi" ]; then
   EXTRA_PARAMS+="--setup_opt multi "
elif [ $MODE == "multi-incr-cll" ]; then
   EXTRA_PARAMS+="--setup_opt multi-incr-cll "
else 
   EXTRA_PARAMS+="--setup_opt cll-er_kd "
fi

echo "EXTRA_PARAMS: "$EXTRA_PARAMS

python tests/test_base_models_main.py --order_lst ${ORDER_LST[$LANG_ORDER]} \
                                      --lt_queue_mode $MODE \
                                      --ltn_model $LTN_MODEL \
                                      --num_decks $NUM_DECKS \
                                      --update_batch_epoch $BATCH_EPOCH \
                                      --update_everything $UPDATE_EVERYTHING \
                                      --max_mem_sz $MAX_MEM_SZ \
                                      --lt_sampling_mode $RANDOM_BASELINE \
                                      $DATASET_PARAMS \
                                      $EXTRA_PARAMS \
                                      --use_k_means