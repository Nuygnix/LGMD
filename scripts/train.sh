#!/bin/bash

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

seeds=(2030)

BATCH_SIZES=(10 8 4)
LEARNING_RATES=(0.00001 0.00002 0.00004)
NODE_HSZs=(500 300)
GOLBAL_HSZs=(500 300)


for bsz in "${BATCH_SIZES[@]}"; do
for lr in "${LEARNING_RATES[@]}"; do
for dim1 in "${NODE_HSZs[@]}"; do
for dim2 in "${GOLBAL_HSZs[@]}"; do
for seed in "${seeds[@]}"; do

    echo "Running with seed $seed"

    lr2=$(awk "BEGIN {print $lr * 2}")
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.run --nproc_per_node 2 $proj_dir/src/main.py \
--do_train \
--use_ddp \
--with_inner_syntax \
--with_global \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size $bsz \
--eval_batch_size 16 \
--epochs 20 \
--logging_steps $((400 / $bsz)) \
--eval_steps $((400 / $bsz)) \
--patience 6 \
--bert_learning_rate $lr \
--learning_rate $lr2 \
--warmup_proportion 0.02 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 18 \
--seed $seed \
--node_hidden_size $dim1 \
--global_hidden_size $dim2

done
done
done
done
done
