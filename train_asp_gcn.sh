#!/bin/bash
# training command for different datasets.

python train_gcn.py --data_dir dataset/Restaurants --vocab_dir dataset/Restaurants --num_layers 8  --seed 80
