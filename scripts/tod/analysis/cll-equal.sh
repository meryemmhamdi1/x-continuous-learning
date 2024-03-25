#!/usr/bin/env bash

MODEL_VER=${1:-"vanilla"} # multi, mono, vanilla, kd-rep, kd-logits, ewc_online, er

echo "CLL-EQUAL "$MODEL_VER

if [ $MODEL_VER == "multi" ]; then
    sh scripts/tod/train_trans_nlu_cll_er_kd.sh \
      "yes" \
      "BertBaseMultilingualCased" \
      "multi-equal"

elif [ $MODEL_VER == "mono" ]; then
    for lang in "en" "de" "fr" "es" "hi" "th"
    do
        sh scripts/tod/train_trans_nlu_cll_er_kd.sh \
          "yes" \
          "BertBaseMultilingualCased" \
          "cll-equal-er_kd" \
          0 \
          "en" \
          0 \
          $lang
    done

else
    if [ $MODEL_VER == "vanilla" ]; then
        SETUP="cll-equal"
    else
        SETUP="cll-equal-er_kd"
    if

    for LANG_ORDER in "en_de_fr_hi_es_th" "th_es_hi_fr_de_en" "es_hi_en_de_th_fr" "fr_th_de_en_hi_es" "hi_en_es_th_fr_de" "de_fr_th_es_en_hi"
    do
      sh scripts/tod/train_trans_nlu_cll_er_kd.sh \
        "yes" \
        "BertBaseMultilingualCased" \
        $SETUP \
        0 \
        "en" \
        0 \
        $LANG_ORDER \
        $MODEL_VER
    done

fi
  



