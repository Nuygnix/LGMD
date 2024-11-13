#!/bin/bash
#SBATCH -J LGMD
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

seeds=(42)


for seed in "${seeds[@]}"
do
    echo "Running with seed $seed"
python $proj_dir/src/main.py  \
--do_train \
--with_global \
--with_inner_syntax \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 250 \
--eval_steps 250 \
--patience 5 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
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
--with_inner_syntax \
--plm_path /public/home/zhouxiabing/data/kywang/plms/bert-base-uncased \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts \
--train_batch_size 4 \
--eval_batch_size 8 \
--epochs 20 \
--logging_steps 250 \
--eval_steps 250 \
--patience 5 \
--bert_learning_rate 1e-5 \
--learning_rate 2e-5 \
--warmup_proportion 0. \
--max_to_save 3 \
--max_seq_len 128 \
--max_node_len 16 \
--seed $seed \
--node_hidden_size1 300 \
--node_hidden_size2 300 \
--global_hidden_size 500
done

