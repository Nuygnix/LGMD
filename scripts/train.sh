#!/bin/bash

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

seeds=(2030)


for seed in "${seeds[@]}"
do
    echo "Running with seed $seed"

python $proj_dir/src/main.py  \
--do_train \
--with_inner_syntax \
--with_global \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 100 \
--eval_steps 100 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 500


python $proj_dir/src/main.py  \
--do_train \
--with_global \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 100 \
--eval_steps 100 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 500

python $proj_dir/src/main.py  \
--do_train \
--with_inner_syntax \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 100 \
--eval_steps 100 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 500

python $proj_dir/src/main.py  \
--do_train \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 100 \
--eval_steps 100 \
--patience 6 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 500


done
