#!/bin/bash

#CUDA_VISIBLE_DEVICES=1 python3 run_seq2seq_attn_val_test.py --dataset_name Math23K --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

#CUDA_VISIBLE_DEVICES=1 python3 run_seq2tree_val_test.py --dataset_name Math23K --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

CUDA_VISIBLE_DEVICES=1 python3 run_lm2seq_mp_val_test.py --dataset_name Math23K --bert_learning_rate 5e-5 --learning_rate 1e-3 --weight_decay 2e-5 --enable_beam_search

CUDA_VISIBLE_DEVICES=1 python3 run_lm2tree_mp_val_test.py --dataset_name Math23K --bert_learning_rate  5e-5 --learning_rate 1e-3 --weight_decay 2e-5 --enable_beam_search

