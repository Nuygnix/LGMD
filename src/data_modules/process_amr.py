#!/usr/bin/env python
# coding: utf-8

import re
import os
import pandas as pd
import pickle
import json

from tqdm import tqdm

import emoji


from transition_amr_parser.parse import AMRParser
from propbank import trans_node

DATA_DIR = "/public/home/zhouxiabing/data/kywang/AMR_MD/data/"


def processTweet(tweet):
    tweet = tweet.lower()
    tweet = emoji.demojize(tweet)
    # tweet = " " + tweet
    tweet=re.sub('thats',"that's",tweet)
    tweet = re.sub('\(@\)', '@', tweet)
    tweet = re.sub(' u ', " you ", tweet)
    tweet = re.sub(' im ', " i'm ", tweet)
    tweet = re.sub('smarter then', "smarter than", tweet)
    tweet = re.sub('isnt', "isn't", tweet)
    tweet = re.sub('youre', "you're", tweet)
    tweet = re.sub('\\\/w', " ", tweet)
    tweet = re.sub(r'[^\x00-\x7F]+','', tweet)
    #tweet = tweet.replace(" rt "," ")
    # tweet = re.sub(' rt ','', tweet)
    tweet = re.sub('(\.)+','.', tweet)
    #tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+) | (http://[^\s]+))','URL',tweet)
    tweet = re.sub('((www\.[^\s]+))','',tweet)
    tweet = re.sub('((http://[^\s]+))','',tweet)
    tweet = re.sub('((https://[^\s]+))','',tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub('_','',tweet)
    tweet = re.sub('\$','',tweet)
    tweet = re.sub('#', ' ', tweet)
    tweet = re.sub('%','',tweet)
    tweet = re.sub('^','',tweet)
    tweet = re.sub('&',' ',tweet)
    tweet = re.sub('\*','',tweet)
    tweet = re.sub('\(','',tweet)
    tweet = re.sub('\)','',tweet)
    tweet = re.sub('-','',tweet)
    tweet = re.sub('«', '', tweet)
    tweet = re.sub('»', '', tweet)
    tweet = re.sub('\+','',tweet)
    tweet = re.sub('=','',tweet)
    tweet = re.sub('"','',tweet)
    tweet = re.sub('~','',tweet)
    tweet = re.sub('`','',tweet)
    # tweet = re.sub('!',' ',tweet)
    # tweet = re.sub(':',' ',tweet)
    tweet = re.sub('^-?[0-9]+$','', tweet)
    tweet = tweet.strip('\'"')
    return tweet




def step1():
    df = pd.read_csv(f'{DATA_DIR}/origin/origin_data.tsv', sep='\t')

    # Download and save a model named AMR3.0 to cache
    parser = AMRParser.from_pretrained('AMR3-structbart-L')

    with open(f'{DATA_DIR}/intermediate/amr_data.jsonl', "w") as f, open(f'{DATA_DIR}/intermediate/amr_error.jsonl', 'w') as f2:
        batch = []
        bsz = 128
        for i in tqdm(range(len(df))):
            number = df['Number'][i]
            turn = df['turn'][i]
            utterance = df['utterance'][i]
            c_r = df['c_r'][i]
            label = df['label'][i]
            clean_text = processTweet(utterance)
            utterance = {
                'number': int(number),
                'turn': int(turn),
                "origin_text": utterance,
                "clean_text": clean_text,
                'label': int(label)
            }
            if c_r == 'rewrite':
                continue

            tokens, positions = parser.tokenize(clean_text)
            utterance['tokens'] = tokens
            utterance['positions'] = positions
            batch.append(utterance)
            if len(batch) == bsz:
                try:
                    _, machines = parser.parse_sentences([item['tokens'] for item in batch], batch_size=bsz)
                except:
                    for utterance in batch:
                        f2.write(json.dumps(utterance) + "\n")
                    batch = []
                    continue
                for j, machine in enumerate(machines):
                    utterance = batch[j]
                    amr = machine.get_amr()
                    utterance['nodes'] = amr.nodes
                    utterance['edges'] = amr.edges
                    utterance['sentence'] = amr.sentence
                    utterance['alignments'] = amr.alignments
                    utterance['root'] = amr.root
                    utterance['roots'] = amr.roots
                    f.write(json.dumps(utterance) + "\n")
                batch = []
        
        if len(batch) != 0:
            try:
                _, machines = parser.parse_sentences([item['tokens'] for item in batch], batch_size=len(batch))
                for j, machine in enumerate(machines):
                    utterance = batch[j]
                    amr = machine.get_amr()
                    utterance['nodes'] = amr.nodes
                    utterance['edges'] = amr.edges
                    utterance['sentence'] = amr.sentence
                    utterance['alignments'] = amr.alignments
                    utterance['root'] = amr.root
                    utterance['roots'] = amr.roots
                    f.write(json.dumps(utterance) + "\n")
            except:
                for utterance in batch:
                    f2.write(json.dumps(utterance) + "\n")


def step2():
    #error数据处理
    cnt = 0
    parser = AMRParser.from_pretrained('AMR3-structbart-L')
    with open(f'{DATA_DIR}/intermediate/amr_error.jsonl', "r") as f, open(f'{DATA_DIR}/intermediate/amr_error1.jsonl', "w") as f2:
        while True:
            line = f.readline()
            if not line:
                break
            cnt += 1
            utterance = json.loads(line)
            try:
                _, machine = parser.parse_sentence(utterance['tokens'])
                amr = machine.get_amr()
                utterance['nodes'] = amr.nodes
                utterance['edges'] = amr.edges
                utterance['sentence'] = amr.sentence
                utterance['alignments'] = amr.alignments
                utterance['root'] = amr.root
                utterance['roots'] = amr.roots
                f2.write(json.dumps(utterance) + "\n")
            except:
                breakpoint()
            if cnt % 30 == 0:
                print(cnt)

def step3():
    # 节点替换成相应定义
    df = pd.read_json(f'{DATA_DIR}/intermediate/amr_data.jsonl', lines=True)
    df.drop(columns=['tokens', 'positions', 'sentence', 'alignments'], inplace=True)

    a = [str(df['number'][i]) + "-" + str(df['turn'][i]) for i in range(len(df))]
    assert len(a) == len(set(a)), "Duplicate data exists!"

    def func(x):
        nodes = x['nodes']
        edges = x['edges']
        node_to_id = {}
        node_list = []
        if len(nodes) == 0:
            breakpoint()
        for i, node in enumerate(nodes.items()):
            node_to_id[node[0]] = i
            node_list.append(trans_node(node[1]))
        
        edge_index = []
        edge_type = []
        for edge in edges:
            edge_index.append([node_to_id[edge[0]], node_to_id[edge[2]]])
            edge_type.append(edge[1])
        
        return node_list, edge_index,  edge_type

    df[['nodes', 'edge_index', 'edge_type']] = df.apply(func, axis=1, result_type='expand')
    df.drop(columns=['edges'], inplace=True)

    df = df.sort_values(by=['number', 'turn']).reset_index(drop=True)
    df = df.groupby('number').agg(list).reset_index()

    # 划分训练集、验证集、测试集
    dev_id_list = pickle.load(open(f'{DATA_DIR}/origin/dev_id_list.pickle', 'rb'))
    test_id_list = pickle.load(open(f'{DATA_DIR}/origin/test_id_list.pickle', 'rb'))
    dev_id_list = [int(id) for id in dev_id_list]
    test_id_list = [int(id) for id in test_id_list]

    dev_df = df[df['number'].isin(dev_id_list)]
    test_df = df[df['number'].isin(test_id_list)]
    train_df = df[~df['number'].isin(dev_id_list + test_id_list)]

    # 保存为 jsonl 文件
    train_df.to_json(f"{DATA_DIR}/final/train_mdrdc.jsonl", orient="records", lines=True)
    dev_df.to_json(f"{DATA_DIR}/final/dev_mdrdc.jsonl", orient="records", lines=True)
    test_df.to_json(f"{DATA_DIR}/final/test_mdrdc.jsonl", orient="records", lines=True)

    print(f"total num of dialogue is {len(df)}")
    print(f"train num of dialogue is {len(train_df)}")
    print(f"dev num of  dialogue is {len(dev_df)}")
    print(f"test num of dialogue is {len(test_df)}")
    


if __name__ == '__main__':
    # step1()
    # step2()
    # with open("/public/home/zhouxiabing/data/kywang/AMR_MD/data/amr_data.jsonl", "a") as f1, open("/public/home/zhouxiabing/data/kywang/AMR_MD/data/amr_error1.jsonl", "r") as f2:
    #     while True:
    #         line = f2.readline()
    #         if not line:
    #             break
    #         f1.write(line)
    step3()
