#!/usr/bin/env bash
. scripts/hyperparam.config
. scripts/paths.config

# TODO ADD HELPER USAGE FUNCTION

TRANS_MODEL=${1:-"BertBaseMultilingualCased"}
STREAM_SETUP=${2:-"cll"}
FREEZE_TYPE=${3:-"none"}
MODEL_EXPANSION_OPT=${4:-"none"}
CONT_LEARN_ALG=${5:-"vanilla"}
TRANS_LAYERS=${6:-"all"}
ORDER_CLASS=${7:-0}
ORDER_LANG=${8:-0}
ORDER_STR=${9:-"en de fr es th"}
SEED=${10:-42}
USE_SLOTS=${11:-"true"}


## Data Stream Setups Options
DATA_OPTIONS=""

if [ $STREAM_SETUP == "cil" ] || [ $STREAM_SETUP == "multi-incr-cil" ] || [ $STREAM_SETUP == "cil-other" ]; then
    DATA_OPTIONS=" --order_class "$ORDER_CLASS
elif [ $STREAM_SETUP == "cll" ] || [ $STREAM_SETUP == "multi-incr-cll" ]; then
    DATA_OPTIONS=" --order_lang "$ORDER_LANG" --order_str "$ORDER_STR
elif [ $STREAM_SETUP == "cil-ll" ]; then
    DATA_OPTIONS="--order_class "$ORDER_CLASS" --order_lang "$ORDER_LANG" --order_str "$ORDER_STR
fi

## Base Model Options
BASE_MODEL_OPTIONS=""
if [ $USE_SLOTS == "yes" ]; then
    BASE_MODEL_OPTIONS=" --use_slots "
fi

CONT_LEARN_OPTIONS=""
## Freezing Options
if [ $FREEZE_TYPE == "freeze_trans" ]; then
    CONT_LEARN_OPTIONS=" --freeze_trans"
elif [ $FREEZE_TYPE == "freeze_linear" ]; then
    CONT_LEARN_OPTIONS=" --freeze_linear"
elif [ $exp_type == "freeze_trans_linear" ]; then
    CONT_LEARN_OPTIONS=" --freeze_trans --freeze_linear"

## Model Expansion Options
if [ $MODEL_EXPANSION_OPT == "adapters" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --use_adapters --adapter_layers "${11}
elif [ $MODEL_EXPANSION_OPT == "multi_head_in" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --multi_head_in --emb_enc_lang_spec "${11}
elif [ $MODEL_EXPANSION_OPT == "multi_head_out" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --multi_head_out"
elif [ $MODEL_EXPANSION_OPT == "multi_head_in_out" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --multi_head_in --multi_head_out --emb_enc_lang_spec "${11}
fi

## Continuous Learning Algorithm Options
if [ $CONT_LEARN_ALG == "ewc" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --cont_learn_alg 'ewc' "
elif [ $CONT_LEARN_ALG == "ewc_online" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --cont_learn_alg 'ewc' --use_online "
elif [ $CONT_LEARN_ALG == "gem" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --cont_learn_alg 'gem' "
elif [ $CONT_LEARN_ALG == "agem" ]; then
    CONT_LEARN_OPTIONS=$CONT_LEARN_OPTIONS" --cont_learn_alg 'gem' --use_a_gem"
fi

python -W ignore main.py $DATA_OPTIONS \
                         $BASE_MODEL_OPTIONS \
                         $CONT_LEARN_OPTIONS \
                         --trans_model $TRANS_MODEL \
                         --setup_opt $STREAM_SETUP \
                         --seed $SEED \
                         --data_root $DATA_ROOT \
                         --out_dir $OUT_DIR \
                         --model_root $MODEL_ROOT \
                         --num_intent_tasks $NUM_INTENT_TASKS \
                         --num_lang_tasks $NUM_LANG_TASKS \
                         --epochs $EPOCHS \
                         --batch_size $BATCH_SIZE \
                         --adam_lr $ADAM_LR\
                         --adam_eps $ADAM_EPS \
                         --beta_1 $BETA_1 \
                         --beta_2 $BETA_2 \
                         --eval_steps $EVAL_STEPS \
                         --no_debug

