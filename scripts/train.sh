#!/bin/bash
proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

# export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES="0,1" nohup python $proj_dir/src/main.py  \
--do_train \
--with_inner_syntax \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 1 \
--eval_batch_size 8 \
--grad_accum_steps 1 \
--epochs 10 \
--logging_steps 300 \
--eval_steps 300 \
--patience 10 \
--warmup_proportion 0.01 \
--bert_learning_rate 2e-5 \
--learning_rate 8e-5 \
--weight_decay 0.01 \
--max_to_save 5 \
--max_seq_len 300 \
--max_node_len 300 \
--seed 42 > output.log 2>&1 &