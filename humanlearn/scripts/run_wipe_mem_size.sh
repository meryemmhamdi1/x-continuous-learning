#!/usr/bin/env bash

TASK_NAME=${1:-"mtop"} # "xnli" "tydiqa" "panx" "multiatis"
MODEL=${2:-"vanilla"} # vanilla, incr-joint, multi, er-rand, mer-rand, {cont-mono,cont-multi}_{fifo,rand,rand-prop}, {cont-mono,cont-multi}_{rand,rand-prop}, {cont-mono,cont-multi}_{fifo,rand,rand-prop}_{er-only,er-main,er-both}_{easy,hard,random,balanced,extreme},{easy,hard,random}
BATCH_EPOCH=${3:-"epoch"} # batch
DECKS=${4:-5}
MAX_MEM_SZ=${5:-10105} # 500 1000 2500 5000 7500

LTN_MODEL="ltn"

for LANG_ORDER in 0 1 2 3
do
    echo "RUNNING "$MODEL" >>>>>>>"
    if [ $MODEL == "vanilla" ]; then
        sh humanlearn/scripts/main_wipe_mem_size.sh $TASK_NAME $LANG_ORDER "cont-mono" "no"
    
    elif [ $MODEL == "incr-joint" ]; then
        sh humanlearn/scripts/main_wipe_mem_size.sh $TASK_NAME $LANG_ORDER "multi-incr-cll" "no"
    
    elif [ $MODEL == "multi" ]; then
        sh humanlearn/scripts/main_wipe_mem_size.sh $TASK_NAME $LANG_ORDER "multi" "no"

    elif [ $MODEL == "er-rand" ]; then
        sh humanlearn/scripts/main_wipe_mem_size.sh $TASK_NAME $LANG_ORDER "cont-mono" "no" $LTN_MODEL "fifo" "yes" 5 "epoch" "er-main" "random"

    elif [ $MODEL == "mer-rand" ]; then
        sh humanlearn/scripts/main_wipe_mem_size.sh $TASK_NAME $LANG_ORDER "cont-mono" "no" $LTN_MODEL "fifo" "yes" 5 "epoch" "er-main" "equal-lang"

    elif [ $MODEL == "cont-mono_fifo" ] || [ $MODEL == "cont-multi_fifo" ] || [ $MODEL == "cont-mono_rand" ] || [ $MODEL == "cont-multi_rand" ] ; then
        OPTIONS=($(echo $MODEL | tr "_" "\n"))  

        MODE=${OPTIONS[0]}
        RANDOM_BASELINE=${OPTIONS[1]}

        USE_LEITNER="yes"
        USE_ER="no"
        
        sh humanlearn/scripts/main_wipe_mem_size.sh \
            $TASK_NAME \
            $LANG_ORDER \
            $MODE \
            $USE_LEITNER \
            $LTN_MODEL \
            $RANDOM_BASELINE \
            $USE_ER \
            $DECKS \
            $BATCH_EPOCH

    else 
        echo "MEM SIZE " $MAX_MEM_SZ
        OPTIONS=($(echo $MODEL | tr "_" "\n"))  

        MODE=${OPTIONS[0]}
        RANDOM_BASELINE=${OPTIONS[1]}
        ER_LQ_SCHEDULER_TYPE=${OPTIONS[2]}
        ER_STRATEGY=${OPTIONS[3]}
        WIPE_STRATEGY=${OPTIONS[4]}

        USE_ER="yes"
        USE_LEITNER="yes"
        
        sh humanlearn/scripts/main_wipe_mem_size.sh \
            $TASK_NAME \
            $LANG_ORDER \
            $MODE \
            $USE_LEITNER \
            $LTN_MODEL \
            $RANDOM_BASELINE \
            $USE_ER \
            $DECKS \
            $BATCH_EPOCH \
            $ER_LQ_SCHEDULER_TYPE \
            $ER_STRATEGY \
            "everything" \
            $WIPE_STRATEGY \
            $MAX_MEM_SZ
    fi
done