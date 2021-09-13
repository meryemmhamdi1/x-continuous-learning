#!/usr/bin/env bash
. scripts/hyperparam.config
. scripts/paths.config
python -W ignore main_ewc.py --data_root $data_root \
                             --no_debug \
                             --out_dir $out_dir \
                             --setup_opt "cll" \
                             --cont_learn_alg "ewc" \
                             --use_online \
                             --order_class 0 \
                             --order_lang 0 \
                             --order_str "hi" "de" "es" "th" "fr" "en" \
                             --trans_model "BertBaseMultilingualCased" \
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
                             --seed 42
