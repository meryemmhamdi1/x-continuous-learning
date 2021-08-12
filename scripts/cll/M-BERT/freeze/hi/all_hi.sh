#!/usr/bin/env bash
echo "Executing HI freezing linear"
sh scripts/cll/M-BERT/freeze/hi/trans_nlu_hi_freeze_linear.sh
echo "Executing HI freezing mbert"
sh scripts/cll/M-BERT/freeze/hi/trans_nlu_hi_freeze_mbert.sh
echo "Executing HI freezing mbert + linear"
sh scripts/cll/M-BERT/freeze/hi/trans_nlu_hi_freeze_mbert_linear.sh