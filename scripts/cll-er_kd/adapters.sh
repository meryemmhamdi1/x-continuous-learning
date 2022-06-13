#!/usr/bin/env bash
for lang_order in "en_de_fr_hi_es_th" "th_es_hi_fr_de_en" "es_hi_en_de_th_fr" "fr_th_de_en_hi_es" "hi_en_es_th_fr_de" "de_fr_th_es_en_hi"
do
  for SEED in 35 40 42
  do
    python -W ignore main_ada.py --languages $lang_order --seed $SEED
    python -W ignore main_ada.py --languages $lang_order --freeze_bert --seed $SEED
  done
done
