import pandas as pd
from tqdm import tqdm


DATA_DIR = "/public/home/zhouxiabing/data/kywang/AMR_MD/data/"

if __name__ == '__main__':
    df = pd.read_json(f"{DATA_DIR}/intermediate/amr_data.jsonl", lines=True)

    # 节点数
    df['num_nodes'] = df['nodes'].apply(len)
    node_counts = df['num_nodes'].value_counts().sort_index()
    length_prob = node_counts / node_counts.sum()
    cumulative_prob = length_prob.cumsum()
    print(cumulative_prob)

    # 边数
    df['num_edges'] = df['edges'].apply(len)
    edges_counts = df['num_edges'].value_counts().sort_index()
    length_prob = edges_counts / edges_counts.sum()
    cumulative_prob = length_prob.cumsum()
    print(cumulative_prob)

    
    # 边类型
    df_edges = pd.DataFrame({'edge_types': [edge[1] for edges in df['edges'] for edge in edges]})

    # 使用 value_counts() 统计各元素的频率
    value_counts = df_edges['edge_types'].value_counts()
    length_prob = value_counts / value_counts.sum()
    cumulative_prob = length_prob.cumsum()
    print(cumulative_prob[:30])

    value_counts.to_csv(f"{DATA_DIR}/final/edge_types_amr.csv", index=True)

    # edge_types = list(set(df_edges['edge_types'].to_list()))
    # with open("/public/home/zhouxiabing/data/kywang/amr_md/data/edge_types_set.txt", "w") as f:
    #     for edge in edge_types:
    #         f.write(edge + "\n")

    # 节点元素
    node_list = []
    for nodes in df['nodes']:
        for node in nodes.values():
            node_list.append(node)
    
    node_list = list(set(node_list))
    with open(f"{DATA_DIR}/intermediate/node_set.txt", "w") as f:
        for node in node_list:
            f.write(node + "\n")

    # 统计话语和节点文本的id长度分布
    df = pd.read_json(f"{DATA_DIR}/final/train_mdrdc.jsonl", lines=True)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("/public/home/zhouxiabing/data/kywang/plms/bert-base-uncased")
    utterance_lens = []
    node_lens = []
    for i in tqdm(range(len(df))):
        turn = len(df.loc[i]['clean_text'])
        for t in range(turn):
            tokenized_text = tokenizer(df.loc[i]['clean_text'][t], truncation=False,
                add_special_tokens=True, return_tensors='pt', return_attention_mask=False
            )
            utterance_lens.append(tokenized_text['input_ids'].shape[1])

            for node in df.loc[i]['nodes'][t]:
                tokenized_text = tokenizer(node, truncation=False,
                    add_special_tokens=True, return_tensors='pt', return_attention_mask=False
                )
                node_lens.append(tokenized_text['input_ids'].shape[1])

    
    df_utterance_lens = pd.DataFrame({'utterance_lens': utterance_lens})
    value_counts = df_utterance_lens['utterance_lens'].value_counts().sort_index()
    length_prob = value_counts / value_counts.sum()
    cumulative_prob = length_prob.cumsum()
    print(cumulative_prob[30:60])


    df_node_lens = pd.DataFrame({'node_lens': node_lens})
    value_counts = df_node_lens['node_lens'].value_counts().sort_index()
    length_prob = value_counts / value_counts.sum()
    cumulative_prob = length_prob.cumsum()
    print(cumulative_prob[:30])