#!/bin/bash
cd "F:/aston/machine-learning/lamp-extractor/src/lamp_extractor/models/mobilenet_v2"
MACHINE=developer@192.168.100.10
DATA_PATH=/home/developer/ml/projects/lamp-extractor-dl/out_corner/mobilenet_v2
MODEL_WEIGTHS_PATH=best_model_weights.pth
TRANSFORM_PATH=infer_transform.yaml
CONFIG_PATH=config.yaml
MODEL_SRC_PATH=model.py

LOADER_SRC_PATH=loader.py
LOGS=logs
INFER_TRANSFORM=infer_transform.yaml

# Download multiple files from remote 
scp $MACHINE:$DATA_PATH/\{$CONFIG_PATH,$LOGS,$TRANSFORM_PATH,$LOADER_SRC_PATH,$MODEL_WEIGTHS_PATH,$MODEL_SRC_PATH\} ./
