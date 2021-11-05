#!/bin/bash

#CUDA_VISIBLE_DEVICES=1 python3 run_mvseq2tree_val_test.py --dataset_name Math23K --learning_rate 1e-3 --weight_decay 1e-5 --use_share_decoder --enable_beam_search

#CUDA_VISIBLE_DEVICES=1 python3 run_mvgraph2tree_val_test.py --dataset_name Math23K --learning_rate 1e-3 --weight_decay 1e-5 --use_share_decoder --enable_beam_search

CUDA_VISIBLE_DEVICES=1 python3 run_mvlm2seq_val_test.py --dataset_name Math23K --bert_learning_rate 5e-5 --learning_rate 1e-3 --weight_decay 2e-5 --use_share_decoder --enable_beam_search --bert_path ./pretrained_lm/chinese-bert-wwm

CUDA_VISIBLE_DEVICES=1 python3 run_mvlm2tree_val_test.py --dataset_name Math23K --bert_learning_rate 5e-5 --learning_rate 1e-3 --weight_decay 2e-5 --use_share_decoder --enable_beam_search --bert_path ./pretrained_lm/chinese-bert-wwm
