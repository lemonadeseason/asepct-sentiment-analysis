#!/bin/bash
# training command for different datasets.

python train_lstm.py --data_dir dataset/Restaurants --vocab_dir dataset/Restaurants --num_layers 2 --seed 29

