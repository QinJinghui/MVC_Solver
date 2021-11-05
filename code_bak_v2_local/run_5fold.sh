#!/bin/bash

python run_seq2seq_attn.py --dataset_name mawps --enable_beam_search

python run_seq2tree.py --dataset_name mawps --enable_beam_search

python run_lm2tree.py --dataset_name mawps --enable_beam_search

