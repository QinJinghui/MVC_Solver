#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python3 run_mvseq2tree.py --dataset_name mawps --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

#CUDA_VISIBLE_DEVICES=0 python3 run_mvgraph2tree.py --dataset_name mawps --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search

CUDA_VISIBLE_DEVICES=0 python3 run_mvlm2seq.py --dataset_name mawps --bert_learning_rate 5e-5 --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search --bert_path ./pretrained_lm/bert-base-uncased

CUDA_VISIBLE_DEVICES=0 python3 run_mvlm2tree.py --dataset_name mawps --bert_learning_rate 5e-5 --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search --bert_path ./pretrained_lm/bert-base-uncased


#CUDA_VISIBLE_DEVICES=0 python3 run_mvlm2tree.py --dataset_name mawps --bert_learning_rate 2e-5 --learning_rate 1e-3 --weight_decay 1e-5 --enable_beam_search --bert_path ./pretrained_lm/bert-base-uncased
