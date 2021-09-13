#!/usr/bin/env bash
echo "Executing TH freezing linear"
sh scripts/cll/M-BERT/freeze/th/trans_nlu_th_freeze_linear.sh
echo "Executing TH freezing mbert"
sh scripts/cll/M-BERT/freeze/th/trans_nlu_th_freeze_mbert.sh
echo "Executing TH freezing mbert + linear"
sh scripts/cll/M-BERT/freeze/th/trans_nlu_th_freeze_mbert_linear.sh