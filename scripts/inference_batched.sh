#!/bin/bash

# step1. set TRAIN_CONFIG path to config file

TRAIN_CONFIG="configs/inference/inference_lam_cafca.yaml"
MODEL_NAME="exps/releases/lam/lam-20k/step_045500/"

TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "MODEL_NAME: $MODEL_NAME"

device=0
nodes=0

export PYTHONPATH=$PYTHONPATH:$pwd

CUDA_VISIBLE_DEVICES=$device python -m lam.launch infer.infer --config $TRAIN_CONFIG model_name=$MODEL_NAME


        
