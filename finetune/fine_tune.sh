#!/usr/bin/env bash
python compress_classifier.py -a vit_oxfordpets -j 4 --epochs 10 -b 8 -p 15 --wd 0.01 --gpus 0 -o logs --deterministic --vs 0.0 --confusion --lr 1e-4 --compress sched.yaml /data/datasets/oxfordpets
