#!/bin/bash
EMOTION=$1
SAVE_PATH=$2

MODEL_PATH=trained_models/blm_"$EMOTION".pt

echo $EMOTION
echo $MODEL_PATH
echo $SAVE_PATH

for DATE in 05-25 05-26 05-27 05-28 05-29 05-30 05-31 06-01 06-02 06-03 06-04 06-05 06-06 06-07 06-08 06-09 06-10 06-11 06-12 06-13 06-14 06-15 06-16 06-17 06-18 06-19 06-20 06-21 06-22 06-23 06-24 06-25 06-26 06-27 06-28 06-29 06-30
do
    echo $DATE
    python src_model/generate_binary.py --bert_path $PATH_BERT --file_path "$PATH_TWEETS/$DATE-original.pkl" --load_from $MODEL_PATH  --target_emotion $EMOTION -np 18 --save_path $SAVE_PATH
done