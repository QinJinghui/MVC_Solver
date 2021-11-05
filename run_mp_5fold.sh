#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python3 run_seq2seq_attn.py --dataset_name mawps --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

# CUDA_VISIBLE_DEVICES=0 python3 run_seq2tree.py --dataset_name mawps --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

CUDA_VISIBLE_DEVICES=0 python3 run_lm2seq_mp.py --dataset_name mawps --bert_learning_rate 5e-5 --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

CUDA_VISIBLE_DEVICES=0 python3 run_lm2tree_mp.py --dataset_name mawps --bert_learning_rate 5e-5 --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

