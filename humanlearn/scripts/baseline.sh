#!/usr/bin/env bash

TASK=${1:-"mtop"} # mtop, xnli, tydiqa, panx
LANG_ORDER=${2:-0} # 0 1 2 3 4 5 6 7
MODE=${3:-"cont-mono"} # "mono", "cont-mono", "cont-multi", "multi", "multi-incr-cll"
MAX_MEM_SZ=${4:-1000} # 100 10105(25%) 20210(50%) 30315(75%) 40421(100%) 
ER_STARTEGY=${5:-"random"} # equal-lang

# Dataset Parameters
DATASET_PARAMS=""
if [ $TASK == "mtop" ]; then
   DATASET_PARAMS+="--task_name tod --data_name mtop --data_format txt --use_slots --use_crf"
   ORDER_LST=("en_de_hi_th" "th_hi_de_en" "hi_th_en_de" "de_en_th_hi" "en" "de" "hi" "th") 
elif [ $TASK == "xnli" ]; then
   DATASET_PARAMS+="--task_name nli --data_name xnli --data_format tsv"
   ORDER_LST=("en_vi_ar_tr" "tr_ar_vi_en" "ar_tr_en_vi" "vi_en_tr_ar" "vi" "en" "tr" "ar")
elif [ $TASK == "tydiqa" ]; then
   DATASET_PARAMS+="--task_name qa --data_name tydiqa --data_format json"
   ORDER_LST=("ru_id_te_sw" "sw_te_id_ru" "te_sw_ru_id" "id_ru_sw_te" "id" "ru" "sw" "te")
elif [ $TASK == "panx" ]; then
   DATASET_PARAMS+="--task_name ner --data_name panx --data_format txt"
   ORDER_LST=("ru_id_te_sw" "sw_te_id_ru" "te_sw_ru_id" "id_ru_sw_te" "id" "ru" "sw" "te")
fi

python humanlearn/test_base_models_main.py --order_lst ${ORDER_LST[$LANG_ORDER]} \
                                      --lt_queue_mode $MODE \
                                      $DATASET_PARAMS \
                                      $EXTRA_PARAMS \
                                      --use_er \
                                      --max_mem_sz $MAX_MEM_SZ \
                                      --setup_opt "cll-er_kd" \
                                      --er_strategy $ER_STARTEGY