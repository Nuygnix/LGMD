#!/bin/bash
#SBATCH -J LGMD
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

python $proj_dir/src/main.py  \
--do_test \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts/2024_bert-2024-11-13-13:55:20
