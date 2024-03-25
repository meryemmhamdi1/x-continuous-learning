#!/usr/bin/env bash

SETUP=${1:-"mono"}  #cont

CONFIG_PARAMS=""
if [ $SETUP == "mono" ]; then
  CONFIG_PARAMS+=" --use_mono"
  LANG_ORDERS=("en" "de" "fr" "hi" "es" "th")
else
  LANG_ORDERS=("en_de_fr_hi_es_th" "th_es_hi_fr_de_en" "es_hi_en_de_th_fr" "fr_th_de_en_hi_es" "hi_en_es_th_fr_de" "de_fr_th_es_en_hi")
fi

for LANG_ORDER in "${LANG_ORDERS[@]}"
do
  for SEED in 35 40 42
  do
    python -W ignore main_ada.py --languages $LANG_ORDER --freeze_bert --seed $SEED $CONFIG_PARAMS
    python -W ignore main_ada.py --languages $LANG_ORDER --seed $SEED $CONFIG_PARAMS
  done
done
