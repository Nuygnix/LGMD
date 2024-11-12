#!/bin/bash
#SBATCH -J AMR
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

seeds=(42 2024 1024)


for seed in "${seeds[@]}"
do
    echo "Running with seed $seed"
python $proj_dir/src/main.py  \
--do_train \
--with_inner_syntax \
--with_global \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 250 \
--eval_steps -1 \
--patience 4 \
--warmup_proportion 0.01 \
--bert_learning_rate 2e-5 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 16 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 300


python $proj_dir/src/main.py  \
--do_train \
--with_inner_syntax \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 250 \
--eval_steps -1 \
--patience 4 \
--warmup_proportion 0.01 \
--bert_learning_rate 2e-5 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 16 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 300


python $proj_dir/src/main.py  \
--do_train \
--with_global \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 250 \
--eval_steps -1 \
--patience 4 \
--warmup_proportion 0.01 \
--bert_learning_rate 2e-5 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 16 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 300
done

