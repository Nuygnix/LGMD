#!/bin/bash
proj_dir=/public/home/zhouxiabing/data/kywang/AMR_MD

export CUDA_LAUNCH_BLOCKING=1

python $proj_dir/src/main.py  \
--do_test \
--ckpt_dir /public/home/zhouxiabing/data/kywang/AMR_MD/ckpts/2080_bert_amr_disc-2024-12-04-03:50:18