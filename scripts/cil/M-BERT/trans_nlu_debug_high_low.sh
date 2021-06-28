#!/usr/bin/env bash
. scripts/hyperparam.config
. scripts/paths.config
python main.py --data-root $data_root \
               --use-slots \
               --verbose \
               --out-dir $out_dir \
               --setup-opt "cil" \
               --order-class 0 \
               --order-lang 0 \
               --verbose \
               --trans-model "BertBaseMultilingualCased" \
               --model-root $model_root \
               --languages "de" "en" "es" "fr" "hi" "th" \
               --num-intent-tasks $num_intent_tasks \
               --num-lang-tasks $num_lang_tasks \
               --epochs $epochs \
               --batch-size $batch_size \
               --adam-lr $adam_lr \
               --adam-eps $adam_eps \
               --beta-1 $beta_1 \
               --beta-2 $beta_2 \
               --eval-steps $eval_steps \
               --seed 42
