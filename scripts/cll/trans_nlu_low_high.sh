#!/usr/bin/env bash
. scripts/hyperparam.config
. scripts/paths.config
python main.py --data-root $data_root \
               --out-dir $out_dir \
               --setup-opt "cll" \
               --trans-model $trans_model \
               --model-root $model_root \
               --use-slots \
               --languages "de" "en" "es" "fr" "hi" "th" \
               --num-intent-tasks $num_intent_tasks \
               --num-lang-tasks $num_lang_tasks \
               --epochs $epochs \
               --batch-size $batch_size \
               --adam-lr $adam_lr \
               --adam-eps $adam_eps \
               --order-class 0 \
               --order-lang 1 \
               --seed 42
