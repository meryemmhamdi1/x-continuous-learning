#!/usr/bin/env bash

USE_SLOTS=${1:-"no"}
TRANS_MODEL=${2:-"BertBaseMultilingualCased"}
STREAM_SETUP=${3:-"cll"}
ORDER_CLASS=${4:-0}
CIL_STREAM_LANG=${5:-"en"}
ORDER_LANG=${6:-0}
ORDER_STR=${7:-"ar_bg_de_el_en_es_fr_hi_ru_sw_th_tr_ur_vi_zh"}
CONT_LEARN_ALG=${8:-"er"}
MODEL_EXPANSION_OPT=${9:-"single_head"}
TRANS_LAYERS=${10:-"all"}
ADAPTER_LAYERS=${11:-"0_1_2_3_4_5_6"}
FREEZE_TYPE=${12:-"none"}
FREEZE_FIRST=${13:-"yes"}
SEED=${14:-42}
CONT_COMP=${15:-"trans gclassifier"}
ADAPTER_TYPE=${16:-"MADX"}
STORING_TYPE=${17:-"ring"}
SAMPLING_TYPE=${18:-"random"}
MAX_MEM_SZ=${19:-6000}
SAMPLING_K=${20:-16}
USE_REPTILE=${21:-"no"}
USE_BATCHES_REPTILE=${22:-"no"}
HYPER_TUNE_IDX=${23:-"0"}
USE_MONO=${24:-"no"}
MONO_LANG=${25:-"en"}
LANGUAGES=${26:-"ar bg de el en es fr hi ru sw th tr ur vi zh"}
USE_PROCESSOR_SHARING=${27:-"yes"}
EVALUATE_ONE_BATCH=${28:-"yes"}
EVAL_SCHED_FREQ=${29:-10}
WARM_START_EPOCHS=${30:-2}
USE_LEITNER_QUEUE=${31:-"no"}
DEMOTE_TO_FIRST_DECK=${32:-"no"}
CONTINUAL_MULTI=${33:-"no"}
FIFO_RAND_MODE=${34:-"fifo"}
LTN_SCHEDULER_TYPE=${35:-"ltn"}
ER_LQ_SCHEDULER_TYPE=${36:-"main"}

## Base Model Options
BASE_MODEL_OPTIONS=""

## Data Stream Setups Options
DATA_OPTIONS=""

if [ $STREAM_SETUP == "cil" ] || [ $STREAM_SETUP == "multi-incr-cil" ] || [ $STREAM_SETUP == "cil-other" ] || [ $STREAM_SETUP == "cll-k-shots" ] ||  [ $STREAM_SETUP == "cll-n-ways" ]; then
    DATA_OPTIONS=" --order_class "$ORDER_CLASS" --order_lst "$ORDER_STR
elif [ $STREAM_SETUP == "cll" ] || [ $STREAM_SETUP == "multi-incr-cll" ] ||  [ $STREAM_SETUP == "cll-er_kd" ] ||  [ $STREAM_SETUP == "cll-equal" ]  ||  [ $STREAM_SETUP == "cll-equal-er_kd" ]; then
    DATA_OPTIONS=" --order_lang "$ORDER_LANG" --order_lst "$ORDER_STR
elif [ $STREAM_SETUP == "cil-ll" ] || [ $STREAM_SETUP == "multi" ]; then
    DATA_OPTIONS="--order_class "$ORDER_CLASS" --order_lang "$ORDER_LANG" --order_lst "$ORDER_STR
fi

###
CONT_LEARN_OPTIONS=""

## Continuous Learning Algorithm Options
if [ $CONT_LEARN_ALG == "ewc" ]; then
    CONT_LEARN_OPTIONS=" --cont_learn_alg ewc "
elif [ $CONT_LEARN_ALG == "ewc_online" ]; then
    CONT_LEARN_OPTIONS=" --cont_learn_alg ewc --use_online "
elif [ $CONT_LEARN_ALG == "gem" ]; then
    CONT_LEARN_OPTIONS=" --cont_learn_alg gem "
elif [ $CONT_LEARN_ALG == "agem" ]; then
    CONT_LEARN_OPTIONS=" --cont_learn_alg gem --use_a_gem"
elif [ $CONT_LEARN_ALG == "er" ] || [ $CONT_LEARN_ALG == "mbpa" ] || [ $CONT_LEARN_ALG == "kd-logits" ] || [ $CONT_LEARN_ALG == "kd-rep" ]; then
    CONT_LEARN_OPTIONS=" --cont_learn_alg "$CONT_LEARN_ALG" --storing_type "$STORING_TYPE" --sampling_type "$SAMPLING_TYPE" --max_mem_sz "$MAX_MEM_SZ" --sampling_k "$SAMPLING_K
fi

## Model Expansion Options
if [ $MODEL_EXPANSION_OPT == "adapters" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --use_adapters --adapter_layers "$ADAPTER_LAYERS
elif [ $MODEL_EXPANSION_OPT == "multi_head_in" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --multi_head_in --emb_enc_subtask_spec "$TRANS_LAYERS
elif [ $MODEL_EXPANSION_OPT == "multi_head_out" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --multi_head_out"
elif [ $MODEL_EXPANSION_OPT == "multi_head_in_out" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --multi_head_in --multi_head_out --emb_enc_subtask_spec "$TRANS_LAYERS
fi

## Freezing Options
if [ $FREEZE_TYPE == "freeze_trans" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --freeze_trans"
elif [ $FREEZE_TYPE == "freeze_linear" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --freeze_linear"
elif [ $exp_type == "freeze_trans_linear" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --freeze_trans --freeze_linear"

fi

META_LEARNING_PARAMS=""
if [ $USE_REPTILE == "yes" ]; then
   META_LEARNING_PARAMS+=" --use_reptile"
fi
if [ $USE_BATCHES_REPTILE == "yes" ]; then
    echo "USE MANY BATCHES REPTILE"
    META_LEARNING_PARAMS+=" --use_batches_reptile"
fi

## MONO Option
USE_MONO_PARAMS=""
if [ $USE_MONO == "yes" ]; then
   USE_MONO_PARAMS+="--use_mono --languages "$MONO_LANG
fi

## LEITNER QUEUES Options
LEITNER_QUEUES_PARAMS=""
if [ $USE_PROCESSOR_SHARING == "yes" ]; then
   LEITNER_QUEUES_PARAMS+="--use_processor_sharing "
fi
if [ $EVALUATE_ONE_BATCH == "yes" ]; then
   LEITNER_QUEUES_PARAMS+="--evaluate_one_batch "
fi
if [ $USE_LEITNER_QUEUE == "yes" ]; then
   LEITNER_QUEUES_PARAMS+="--use_leitner_queue "
fi
if [ $DEMOTE_TO_FIRST_DECK == "yes" ]; then
   LEITNER_QUEUES_PARAMS+="--demote_to_first_deck "
fi
if [ $CONTINUAL_MULTI == "yes" ]; then
   LEITNER_QUEUES_PARAMS+="--use_cont_leitner_queue "
fi

LEITNER_QUEUES_PARAMS+="--eval_sched_freq "$EVAL_SCHED_FREQ" --warm_start_epochs "$WARM_START_EPOCHS

echo "Training " $TRANS_MODEL $STREAM_SETUP

echo "    Data Options " $DATA_OPTIONS
echo "    Base Model Options " $BASE_MODEL_OPTIONS
echo "    Continuous Learn Options " $CONT_LEARN_ALG $CONT_LEARN_OPTIONS
echo "    Model Expansion Options:" $MODEL_EXPANSION_OPT
echo "    Leitner Queues Options:" $LEITNER_QUEUES_PARAMS

python3.6 -W ignore main.py --trans_model $TRANS_MODEL \
                            --task_name "nli"  --data_format "tsv" --data_name "xnli" \
                            --setup_opt $STREAM_SETUP \
                            --seed $SEED \
                            --cont_comp $CONT_COMP \
                            --adapter_type $ADAPTER_TYPE \
                            --cil_stream_lang $CIL_STREAM_LANG \
                            --param_tune_idx $HYPER_TUNE_IDX \
                            --save_model \
                            --save_dev_pred \
                            --save_test_every_epoch \
                            --languages $LANGUAGES \
                            --lt_sampling_mode $FIFO_RAND_MODE \
                            --er_lq_scheduler_type $ER_LQ_SCHEDULER_TYPE \
                            --ltn_scheduler_type $LTN_SCHEDULER_TYPE \
                            $DATA_OPTIONS $BASE_MODEL_OPTIONS $CONT_LEARN_OPTIONS $META_LEARNING_PARAMS \
                            $USE_MONO_PARAMS $LEITNER_QUEUES_PARAMS