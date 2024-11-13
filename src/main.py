import os
import argparse
from utils.tools import *
from models.model import AMRModel
from trainer import Trainer
from transformers import BertConfig, BertTokenizer
from data_modules.data_loader import get_dataloader
from callback.earlystopping import EarlyStopping
import time
import torch
import logging
import os
import json


def run_train(args):
    now_time = f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}"
    model_type = "bert"
    if args.with_inner_syntax:
        model_type += "_amr"
    if args.with_global:
        model_type += "_disc"
    logger = config_logging("train", f"{args.log_dir}/train/{args.seed}_{model_type}-{now_time}.log",
                   logging.INFO, logging.DEBUG)
    print_config(args, logger)
    # load model and tokenizer
    logger.info("Initiate Model and Tokenizer...")

    tokenizer = BertTokenizer.from_pretrained(args.plm_path)

    model = AMRModel(args, 18)
    print_model(model, logger)

    # load data
    train_dataloader = get_dataloader(tokenizer, args.dataset_dir, 'train',
                    args.train_batch_size, args.max_seq_len, args.max_node_len, logger)
    dev_dataloader = get_dataloader(tokenizer, args.dataset_dir, 'dev',
                    args.eval_batch_size, args.max_seq_len, args.max_node_len, logger)
    test_dataloader = get_dataloader(tokenizer, args.dataset_dir, 'test',
                    args.eval_batch_size, args.max_seq_len, args.max_node_len, logger)
    # train
    ckpt_dir = f"{args.ckpt_dir}/{args.seed}_{model_type}-{now_time}"
    early_stopping = EarlyStopping(ckpt_dir, patience=args.patience, mode='max', max_to_save=args.max_to_save)
    
    # save configuration
    os.makedirs(ckpt_dir)
    args_dict = vars(args)
    with open(f'{ckpt_dir}/args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # train
    trainer = Trainer(model, tokenizer, args, logger, early_stopping=early_stopping)
    trainer.train(train_dataloader, dev_dataloader, test_dataloader)



def run_test(args):
    logger = config_logging("test", f"{args.log_dir}/test/{args.ckpt_dir.split('/')[-1]}.log",
                   logging.INFO, logging.DEBUG)
    
    print_config(args, logger)
    logger.info("Load Data and Tokenizer...")

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.plm_path)

    # load data
    test_dataloader = get_dataloader(tokenizer, args.dataset_dir, 'test',
                    args.eval_batch_size, args.max_seq_len, args.max_node_len, logger)
    
    model = AMRModel(args, 18)
    print_model(model, logger)

    trainer = Trainer(model, tokenizer, args, logger)
    for model_name in os.listdir(args.ckpt_dir):
        if '.json' in model_name:
            continue
        checkpoint = torch.load(os.path.join(args.ckpt_dir, model_name))
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Load model state dict from {args.ckpt_dir}/{model_name}")

        trainer.eval(test_dataloader, do_save=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The arguments of Conversation")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')

    # model type
    parser.add_argument("--with_inner_syntax", action='store_true')
    parser.add_argument("--with_global", action='store_true')

    # amr
    parser.add_argument("--num_relations", type=int, default=116) # TODO: 搞个关系列表，自动算
    parser.add_argument("--node_hidden_size1", type=int, default=200)
    parser.add_argument("--node_hidden_size2", type=int, default=200)
    parser.add_argument("--node_hidden_size", type=int, default=200)
    parser.add_argument("--num_node_gnn_layer", type=int, default=1)

    # discourse
    parser.add_argument("--num_global_relations", type=int, default=16)
    parser.add_argument("--num_gloabl_layers", type=int, default=2)    
    parser.add_argument("--global_hidden_size", type=int, default=200)

    # clf
    parser.add_argument("--clf_hidden_size", type=int, default=300)
    parser.add_argument("--num_clf_layers", type=int, default=1)

    # model dir
    parser.add_argument("--plm_path", type=str, default="/public/home/zhouxiabing/data/kywang/plms/bert-base-uncased")
    parser.add_argument("--ckpt_dir", type=str, default="/public/home/zhouxiabing/data/kywang/AMR_MD/ckpts")

    # other dir
    parser.add_argument("--log_dir", type=str, default="/public/home/zhouxiabing/data/kywang/AMR_MD/log")
    parser.add_argument("--test_results_dir", type=str, default="/public/home/zhouxiabing/data/kywang/AMR_MD/test_results")

    # dataset dir
    parser.add_argument("--dataset_dir", type=str, default="/public/home/zhouxiabing/data/kywang/AMR_MD/data/final")

    # data process
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_node_len", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # train
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--logging_steps", type=int, default=400)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad_accum_steps", default=1, type=int)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor"])
    parser.add_argument("--bert_learning_rate", type=float, default=2e-5)
    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", default=1, type=float)

    # 最多保存多少个ckpt
    parser.add_argument("--max_to_save", type=int, default=1)

    # other
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()


    if args.do_train:
        seed_everything(args.seed)
        run_train(args)
    elif args.do_test:
        with open(f"{args.ckpt_dir}/args.json", 'rt') as f:
            json_dict = json.load(f)
            json_dict = {k: v for k, v in json_dict.items() if k not in ['do_train', 'do_test', 'ckpt_dir']}

            argparse_dict = vars(args)
            argparse_dict.update(json_dict)

        seed_everything(args.seed)
        run_test(args)
    
    

