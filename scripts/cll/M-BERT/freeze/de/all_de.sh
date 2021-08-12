#!/usr/bin/env bash
echo "Executing DE freezing linear"
sh scripts/cll/M-BERT/freeze/de/trans_nlu_de_freeze_linear.sh
echo "Executing DE freezing mbert"
sh scripts/cll/M-BERT/freeze/de/trans_nlu_de_freeze_mbert.sh
echo "Executing DE freezing mbert + linear"
sh scripts/cll/M-BERT/freeze/de/trans_nlu_de_freeze_mbert_linear.sh