#!/bin/bash
emotion=$1
BERT_MODEL=$2

for seed in 15 # change this to use different random seeds you want
do
    if [[ "$BERT_MODEL" = "" ]]
    then
        python src_model/main.py -v -s --source go  --target hurricane --suffix hurricane_hg_binary_none_"$emotion"_"$seed" --few_shot --seed $seed --target_emotion $emotion --model_path trained_models/ --summary_path model_summary/ --step_joint 3000 --epoch 1  -dp 0.1 --ddp 0.5 -salt
    else
        python src_model/main.py -v -s --source go --target hurricane --suffix hurricane_hg_binary_"$BERT_MODEL"_"$emotion"_"$seed" --few_shot --bert_path_model $BERT_MODEL --seed $seed --target_emotion $emotion --model_path trained_models/  --summary_path model_summary/ --step_joint 3000 --epoch 1  -dp 0.1 --ddp 0.5 -salt
    fi
done