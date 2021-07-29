#!/usr/bin/env bash
. scripts/hyperparam.config
. scripts/paths.config
python main.py --data_root $data_root \
               --out_dir $out_dir \
               --setup_opt "cil" \
               --order_class 1 \
               --order_lang 0 \
               --trans_model "BertBaseMultilingualCased" \
               --model_root $model_root \
               --use_slots \
               --languages "de" "en" "es" "fr" "hi" "th" \
               --num_intent_tasks $num_intent_tasks \
               --num_lang_tasks $num_lang_tasks \
               --epochs $epochs \
               --batch_size $batch_size \
               --adam_lr $adam_lr \
               --adam_eps $adam_eps \
               --beta_1 $beta_1 \
               --beta_2 $beta_2 \
               --eval_steps $eval_steps \
               --no_debug \
               --seed 42
