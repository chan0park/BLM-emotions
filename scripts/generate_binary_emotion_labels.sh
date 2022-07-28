#!/bin/bash
EMOTION=$1
LOAD_FROM=$2
PATH_INPUT=$3
PATH_OUTPUT=$4

python src_model/infer.py -v -s --input_file $PATH_INPUT --save_output_path $PATH_OUTPUT --load_from $LOAD_FROM --target_emotion $EMOTION --bert_path_model blm