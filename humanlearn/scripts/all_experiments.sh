#!/usr/bin/env bash

LANG=${1:-"en"} # "en_vi_ar_tr_bg_el_ur" "vi_ur_en_ar_tr_bg_el" "ar_en_el_ur_vi_tr_bg" "tr_ar_bg_el_ur_en_vi" 
#"bg_tr_vi_en_el_ur_ar" "el_bg_ur_vi_en_ar_tr" "ur_el_tr_bg_ar_vi_en" 
MODE=${2:-"mono"}

# Baseline
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE


# TOD
python humanlearn/test_base_models_main.py --order_lst $LANG --lt_queue_mode $MODE --use_slots --task_name "tod" --data_name "mtop" --data_format "tsv" 

# Use Leitner Queues
## with default hyperparameters
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner
## demote first
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --demote_to_first_deck
## random
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --lt_sampling_mode "rand"
## RBF
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --ltn_model "rbf"
## Experimenting with updating at the end of each batch vs epoch
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --update_batch_epoch "epoch"
## Experimenting with different numbers of decks 
# python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --n_decks 2
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --num_decks 3
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --num_decks 4
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --num_decks 6
python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --num_decks 7
# python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --n_decks 8
# python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --n_decks 9
# python humanlearn/test_nli_base_models.py --order_lst $LANG --lt_queue_mode $MODE --use_leitner --n_decks 10



