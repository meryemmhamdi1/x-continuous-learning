#!/usr/bin/env bash
. scripts/hyperparam.config
. scripts/paths.config
python -W ignore main.py --data_root $data_root \
                         --no_debug \
                         --freeze_linear \
                         --out_dir $out_dir \
                         --setup_opt "cll" \
                         --order_class 0 \
                         --order_lang 0 \
                         --trans_model "BertBaseMultilingualCased" \
                         --order_str "de" "en" "fr" "hi" "es" "th" \
                         --model_root $model_root \
                         --use_slots \
                         --num_intent_tasks $num_intent_tasks \
                         --num_lang_tasks $num_lang_tasks \
                         --epochs $epochs \
                         --batch_size $batch_size \
                         --adam_lr $adam_lr \
                         --adam_eps $adam_eps \
                         --beta_1 $beta_1 \
                         --beta_2 $beta_2 \
                         --eval_steps $eval_steps \
                         --cont_learn_alg "ewc" \
                         --seed 42