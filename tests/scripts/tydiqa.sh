#!/usr/bin/env bash

python tests/test_base_models_main.py --order_lst "en_ru_ar_id_fi_bn_te_sw" --lt_queue_mode "cont-mono" --task_name "qa" --data_name "tydiqa" --data_format "json"