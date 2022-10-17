#!/usr/bin/env bash

helpFunction()
{
   echo "Multi-purpose program to run different versions/setups of continuous learning"
   echo "Usage: $0 -a USE_SLOTS -b TRANS_MODEL -c STREAM_SETUP -d ORDER_CLASS -e ORDER_LANG -f ORDER_LST -g CONT_LEARN_ALG\
                -h MODEL_EXPANSION_OPT -i TRANS_LAYERS -j ADAPTER_LAYERS -k FREEZE_TYPE -l FREEZE_FIRST\
                and -m SEED"
   echo -e "\t-a USE_SLOTS:\
                * yes (DEFAULT): use slots in the base model.\
                * no: only train for intent detection."
   echo -e "\t-b TRANS_MODEL is the transformers model used which could be: BertBaseMultilingualCased (DEFAULT), BertLarge, \
             BertBaseCased, Xlnet_base , Xlnet_large, XLM, DistilBert_base, DistilBert_large, Roberta_base, Roberta_large,
             XLMRoberta_base, XLMRoberta_large, ALBERT-base-v1 ALBERT-large-v1, ALBERT-xlarge-v1, ALBERT-xxlarge-v1,
             ALBERT-base-v2, ALBERT-large-v2, ALBERT-xlarge-v2, ALBERT-xxlarge-v2"
   echo -e "\t-c STREAM_SETUP: the data stream setup used: \
                    * cil:       Cross-CIL with fixed LL.
                    * cil-other:  Incremental version of cil where previous intents'subtasks are added in addition \
                                  to other labels for subsequent intents'subtasks.
                    * cll (DEFAULT):        Cross-LL with fixed CIL.
                    * cil-ll:     Cross CIL and CLL mixed.
                    * multi-incr-cil: Weaker version of Multi-Task Learning, where we gradually fine-tune on the \
                                      accumulation of different subtasks.
                    * multi-incr-cll: Weaker version of Multilingual Learning, where we gradually fine-tune on the \
                                      accumulation of different languages.
                    * multi:      Multi-tasking one model on all tasks and languages."
   echo -e "\t-d ORDER_CLASS: Different ways of ordering the classes to be seen in the continuous learning: \
                    * 0 (DEFAULT): high2lowclass: decreasing order (from high to low-resource).\
                    * 1: low2highclass: increasing order (from low to high-resource).\
                    * 2: randomclass: random order."
   echo -e "\t-e CIL_STREAM_LANG: Which lang to work on for the CIL setup if it is picked."
   echo -e "\t-f ORDER_LANG: Different ways of ordering the languages:\
                    * 0 (DEFAULT): high2lowlang: decreasing order (from high to low-resource).\
                    * 1: low2highlang: increasing order (from low to high-resource).\
                    * 2: randomlang: random order."
   echo -e "\t-g ORDER_STR: Specific order for subtasks and languages: list of languages/subtasks comes as a list"
   echo -e "\t-h CONT_LEARN_ALG: The continuous learning algorithm used (otherwise) just vanilla tuning\
                    * vanilla (DEFAULT): just vanilla fine-tuning with no continuous learning algorithm.\
                    * ewc \
                    * gem \
                    * mbpa \
                    * metambpa"
   echo -e "\t-i MODEL_EXPANSION_OPT: \
                    * adapters: use adapters\
                    * multi_head_in: use multiple heads that would imply multiple subtask/language-specific heads \
                                     at the input level.\
                    * multi_head_out: use multiple heads in the outputs that would imply the use of different \
                                      task-specific layers.\
                    * multi_head_in_out: use multiple heads at the input and output levels.\
                    * single_head (DEFAULT): single headed on both the input and output."
   echo -e "\t-j TRANS_LAYERS: one or a group of layers in the embeddings or the encoder to tune for each language/subtask independently.\
                    * embeddings \
                    * encoder.layer.0.
                    * encoder.layer.1.
                    * encoder.layer.2.
                    * encoder.layer.3.
                    * encoder.layer.4.
                    * encoder.layer.5.
                    * encoder.layer.6.
                    * encoder.layer.7.
                    * encoder.layer.8.
                    * encoder.layer.9.
                    * encoder.layer.10.
                    * encoder.layer.11.
                    * pooler
                    * all (DEFAULT)"

   echo -e "\t-k ADAPTER_LAYERS: List of integers denoting which decoder layers which adapters are applied."

   echo -e "\t-l FREEZE_TYPE: the freezing mode specifying the layer to be frozen which can be:\
                    * freeze_trans: Whether to freeze all layers in the Transformer, \
                    * freeze_linear: Whether to freeze all task-specific layers.\
                    * freeze_trans_linear: Whether to freeze both transformer.\
                    * none: tune everything transformers and linear."
   echo -e "\t-m FREEZE_FIRST: \
                    * yes: freeze from the first subtask/language.\
                    * no: freeze from the second subtask/language."

   echo -e "\t-n SEED: Random seed for random, numpy and to initialize pytorch weights."

   echo -e "\t-o CONT_COMP: Which component(s) in the model to focus on while learning and during regularization \
            or replay. It can be any combination of the three: \
                     * trans: regularize or replay on transformers encoder only .\
                     * intent: regularize or replay on the intent prediction head only. \
                     * slot: regularize or replay on the slot prediction head only."

    echo -e "\t-p ADAPTER_TYPE: Which adapter architecture:\
                     * Houlsby: resembling the original adapter architecture\
                     * MADX: just one bottleneck block."

   exit 1 # Exit script after printing help
}

while getopts "a:b:c:d:e:f:g:h:i:j:k:l:m:n" opt
do
   case "$opt" in
      a ) USE_SLOTS="$OPTARG" ;;
      b ) TRANS_MODEL="$OPTARG" ;;
      c ) STREAM_SETUP="$OPTARG" ;;
      d ) ORDER_CLASS="$OPTARG" ;;
      e ) CIL_STREAM_LANG="$OPTARG" ;;
      f ) ORDER_LANG="$OPTARG" ;;
      g ) ORDER_STR="$OPTARG" ;;
      h ) CONT_LEARN_ALG="$OPTARG" ;;
      i ) MODEL_EXPANSION_OPT="$OPTARG" ;;
      j ) TRANS_LAYERS="$OPTARG" ;;
      k ) ADAPTER_LAYERS="$OPTARG" ;;
      l ) FREEZE_TYPE="$OPTARG" ;;
      m ) FREEZE_FIRST="$OPTARG" ;;
      n ) SEED="$OP TARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

USE_SLOTS=${1:-"yes"}
TRANS_MODEL=${2:-"BertBaseMultilingualCased"}
STREAM_SETUP=${3:-"cll-er_kd"}
ORDER_CLASS=${4:-0}
CIL_STREAM_LANG=${5:-"en"}
ORDER_LANG=${6:-0}
ORDER_STR=${7:-"en_de_fr_hi_es_th"}
CONT_LEARN_ALG=${8:-"er"}
MODEL_EXPANSION_OPT=${9:-"single_head"}
TRANS_LAYERS=${10:-"all"}
ADAPTER_LAYERS=${11:-"0_1_2_3_4_5_6"}
FREEZE_TYPE=${12:-"none"}
FREEZE_FIRST=${13:-"yes"}
SEED=${14:-42}
CONT_COMP=${15:-"trans intent slot"}
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
LANGUAGES=${26:-"de en es fr hi th"}
USE_PROCESSOR_SHARING=${27:-"yes"}
EVALUATE_ONE_BATCH=${28:-"yes"}
EVAL_SCHED_FREQ=${29:-10}
WARM_START_EPOCHS=${30:-2}
USE_LEITNER_QUEUE=${31:-"no"}
DEMOTE_TO_FIRST_DECK=${32:-"no"}

## Base Model Options
BASE_MODEL_OPTIONS=""
if [ $USE_SLOTS == "yes" ]; then
    BASE_MODEL_OPTIONS=" --use_slots "
fi

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

LEITNER_QUEUES_PARAMS+="--eval_sched_freq "$EVAL_SCHED_FREQ" --warm_start_epochs "$WARM_START_EPOCHS

echo "Training " $TRANS_MODEL $STREAM_SETUP

echo "    Data Options " $DATA_OPTIONS
echo "    Base Model Options " $BASE_MODEL_OPTIONS
echo "    Continuous Learn Options " $CONT_LEARN_ALG $CONT_LEARN_OPTIONS
echo "    Model Expansion Options:" $MODEL_EXPANSION_OPT
echo "    Leitner Queues Options:" $LEITNER_QUEUES_PARAMS

python3.6 -W ignore main_sanity_check_er_kd.py --trans_model $TRANS_MODEL \
                                               --setup_opt $STREAM_SETUP \
                                               --seed $SEED \
                                               --cont_comp $CONT_COMP \
                                               --adapter_type $ADAPTER_TYPE \
                                               --cil_stream_lang $CIL_STREAM_LANG \
                                               --param_tune_idx $HYPER_TUNE_IDX \
                                               --random_pred \
                                               --save_model \
                                               --save_dev_pred \
                                               --languages $LANGUAGES \
                                               $DATA_OPTIONS $BASE_MODEL_OPTIONS $CONT_LEARN_OPTIONS $META_LEARNING_PARAMS \
                                               $USE_MONO_PARAMS $LEITNER_QUEUES_PARAMS