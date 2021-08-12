#!/usr/bin/env bash
echo "Executing ES freezing linear"
sh scripts/cll/M-BERT/freeze/es/trans_nlu_es_freeze_linear.sh
echo "Executing ES freezing mbert"
sh scripts/cll/M-BERT/freeze/es/trans_nlu_es_freeze_mbert.sh
echo "Executing ES freezing mbert + linear"
sh scripts/cll/M-BERT/freeze/es/trans_nlu_es_freeze_mbert_linear.sh