#!/usr/bin/env bash

TASK_NAME=${1:-"mtop"} # "xnli" "tydiqa" "panx"

for LANG_ORDER in 4 5 6 7
do
    echo "RUNNING MONO on "$LANG_ORDER
    sh humanlearn/scripts/main.sh $TASK_NAME $LANG_ORDER "cont-mono" "no"

done