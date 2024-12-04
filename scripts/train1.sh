#!/bin/bash

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

seeds=(2080 42 2024 2025 2001 1217 1115 1001)

for seed in "${seeds[@]}"; do

    echo "Running with seed $seed"

CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--with_inner_syntax \
--with_global \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 200 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--weight_decay 0.01 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500


CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--with_global \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 200 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--weight_decay 0.01 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500



CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--with_inner_syntax \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 200 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--weight_decay 0.01 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500


CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 200 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--weight_decay 0.01 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500

done
