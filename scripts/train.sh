#!/bin/bash

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

seeds=(1001 1115 1217 2001 2024 2025 2030 2080)
BATCH_SIZES=(4 8)
LEARNING_RATES=(0.00001 0.00002 0.00004 0.00008)


for seed in "${seeds[@]}"; do
for bsz in "${BATCH_SIZES[@]}"; do
for lr in "${LEARNING_RATES[@]}"; do

    echo "Running with seed $seed"
    lr2=$(awk "BEGIN {print 2 * $lr}")

CUDA_VISIBLE_DEVICES="0" python $proj_dir/src/main.py \
--do_train \
--with_inner_syntax \
--with_global \
--dataset_name toxichat \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size $bsz \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps 200 \
--eval_steps $((400 / $bsz)) \
--patience 6 \
--bert_learning_rate $lr \
--learning_rate $lr2 \
--warmup_proportion 0.02 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size 500 \
--global_hidden_size 500

done
done
done
