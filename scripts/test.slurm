#!/bin/bash
#SBATCH -J AMR
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

python $proj_dir/src/main.py  \
--do_test \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts/42_bert-2024-11-10-16:31:46 \
--train_batch_size 4 \
--eval_batch_size 8 \
--max_seq_len 128 \
--max_node_len 16 \
--seed 42 \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--node_hidden_size 300

