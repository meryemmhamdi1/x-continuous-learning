#!/usr/bin/env bash
for lang_order in "en" "de" "fr" "hi" "es" "th"
do
  for SEED in 35 40 42
  do
    python -W ignore main_ada.py --use_mono --languages $lang_order --freeze_bert --seed $SEED
    python -W ignore main_ada.py --use_mono --languages $lang_order --seed $SEED
  done
done
