#!/usr/bin/env bash

MODEL_OPT=${1:"vanilla"} # joint multi vanilla er kd-logits kd-rep ewc_online multi_head_in_all multi_head_out multi_head_in_enc8

for LANG_ORDER in "en_th" "fr_es" "th_en" "es_fr" "hi_de" "de_hi"
do
    if [ $LANGUAGE_STR == "en_th" ]; then
        LANGUAGES="en th"
    elif [ $LANGUAGE_STR == "fr_es" ]; then
        LANGUAGES="fr es"
    elif [ $LANGUAGE_STR == "th_en" ]; then
        LANGUAGES="th en"
    elif [ $LANGUAGE_STR == "es_fr" ]; then
        LANGUAGES="es fr"
    elif [ $LANGUAGE_STR == "hi_de" ]; then
        LANGUAGES="hi de"
    elif [ $LANGUAGE_STR == "de_hi" ]; then
        LANGUAGES="de hi"
    fi

    if [ $MODEL_VER == "joint" ]; then
        SETUP_OPT="multi-incr-cll"
        MODEL="vanilla"
        MODEL_EXP="single_head"
        COMPONENTS="all"
    elif [ $MODEL_VER == "multi" ]; then
        SETUP_OPT="multi"
        MODEL="vanilla"
        MODEL_EXP="single_head"
        COMPONENTS="all"
    elif [ $MODEL_VER == "multi_head_in_all" ]; then
        SETUP_OPT="cll-er_kd"
        MODEL="vanilla"
        MODEL_EXP="multi_head_in"
        COMPONENTS="embeddings_encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8._encoder.layer.9._encoder.layer.10._encoder.layer.11._pooler"
    elif [ $MODEL_VER == "multi_head_in_enc8" ]; then
        SETUP_OPT="cll-er_kd"
        MODEL="vanilla"
        MODEL_EXP="multi_head_in"
        COMPONENTS="encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8."
    elif [ $MODEL_VER == "multi_head_out" ]; then
        SETUP_OPT="cll-er_kd"
        MODEL="vanilla"
        MODEL_EXP="multi_head_out"
        COMPONENTS="all"
    else
        SETUP_OPT="cll-er_kd"
        MODEL=$MODEL_VER
        MODEL_EXP="single_head"
        COMPONENTS="all"
    fi

    echo $MODEL_VER" OVER order "$LANG_ORDER
    sh scripts/tod/train_trans_nlu_cll_er_kd.sh \
        "yes" \
        "BertBaseMultilingualCased" \
        $SETUP_OPT \
        0 \
        "en" \
        0 \
        $LANG_ORDER \
        $MODEL \
        $MODEL_EXP \
        $COMPONENTS \
        "0_1_2_3_4_5_6" \
        "none" \
        "yes" \
        42 \
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
        $LANGUAGES
    
    # Missing: adapters
done
