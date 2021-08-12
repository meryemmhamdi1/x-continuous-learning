#!/usr/bin/env bash
echo "Executing FR freezing linear"
sh scripts/cll/M-BERT/freeze/fr/trans_nlu_fr_freeze_linear.sh
echo "Executing FR freezing mbert"
sh scripts/cll/M-BERT/freeze/fr/trans_nlu_fr_freeze_mbert.sh
echo "Executing FR freezing mbert + linear"
sh scripts/cll/M-BERT/freeze/fr/trans_nlu_fr_freeze_mbert_linear.sh