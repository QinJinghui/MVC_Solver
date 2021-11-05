#!/bin/bash

python run_seq2seq_attn_val_test.py --dataset_name Math23K --enable_beam_search

python run_seq2tree_val_test.py --dataset_name Math23K --enable_beam_search

python run_lm2tree_val_test.py --dataset_name Math23K --enable_beam_search

