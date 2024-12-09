#!/bin/bash

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

seeds=(1001 1115 1217 2001 2025 2024 42 2080 2030)

for seed in "${seeds[@]}"; do

    echo "Running with seed $seed"

CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--with_inner_syntax \
--with_global \
--dataset_name toxichat \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 100 \
--patience 5 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0.02 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500


CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--with_inner_syntax \
--dataset_name toxichat \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 100 \
--patience 5 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0.02 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500



CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--with_global \
--dataset_name toxichat \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 100 \
--patience 5 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0.02 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500


CUDA_VISIBLE_DEVICES="1" python $proj_dir/src/main.py \
--do_train \
--dataset_name toxichat \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps 100 \
--patience 5 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0.02 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500

done
