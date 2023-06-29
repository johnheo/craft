#!/bin/bash

export CUDA_VISIBLE_DEVICES=1 # 1,3

# TASK='cifar10'
# TASK='cifar100'
# TASK='flowers-102'
# TASK='stanford-cars'
# TASK='oxford-pets'
TASK='aircraft'
# TASK='food-101'

SAM='sam'

echo "\n\n********Quantizing with $SAM on $TASK"
python example/test_all.py \
    --model vit_base_patch16_224 \
    --resume vit-B-$TASK-$SAM.pt \
    --dataset $TASK \
    # --eval

# dataset options: cifar10, cifar100, flowers-102

