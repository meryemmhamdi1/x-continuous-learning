#!/usr/bin/env bash
. scripts/hyperparam.config
. scripts/paths.config
python main.py --data-root $data_root \
               --out-dir $out_dir \
               --setup-opt "cil-ll" \
               --setup-3 "intents" \
               --order-class 2 \
               --order-lang 2 \
               --trans-model "BertBaseMultilingualCased" \
               --model-root $model_root \
               --use-slots \
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
