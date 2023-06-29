#!/bin/bash

# for s in 0.1 0.2 0.3 0.4 0.5;
# do
#     echo "\n\n********Pruning with sparsity $s"
#     python main.py \
#         --model vit_base_patch16_224 \
#         --data_path cifar100 \
#         --nb_classes 100 \
#         --imagenet_default_mean_and_std False \
#         --resume ckpts/vit-B-sam.pt \
#         --prune_metric wanda \
#         --prune_granularity layer \
#         --sparsity $s
# done

# granularity: layer ; row; 
# for type in "8:16" "4:8" "2:4";
# do
#     echo "\n\n********Pruning with $type sparsity 0.5"
#     python main.py \
#         --model vit_base_patch16_224 \
#         --data_path cifar100 \
#         --nb_classes 100 \
#         --imagenet_default_mean_and_std False \
#         --resume ckpts/vit-B-nosam.pt \
#         --prune_granularity layer \
#         --sparsity_type $type \
#         --prune_metric wanda \
#         --sparsity 0.5
# done

for type in "32:64" "16:32" "8:16";
do
    echo "\n\n********Pruning with $type sparsity 0.5"
    python main.py \
        --model vit_base_patch16_224 \
        --data_path cifar100 \
        --nb_classes 100 \
        --imagenet_default_mean_and_std False \
        --resume ckpts/vit-B-nosam.pt \
        --prune_granularity layer \
        --sparsity_type $type \
        --prune_metric wanda \
        --sparsity 0.5
done