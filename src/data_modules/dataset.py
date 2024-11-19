from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
import torch


class AMRDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, max_seq_len, max_node_len, split, logger):
        super().__init__()
        dataset_path = f"{dataset_dir}/{split}_mdrdc.jsonl"
        self.df = pd.read_json(dataset_path, lines=True)

        logger.info(f"load data from {dataset_path}")
        logger.info(f"total number of dialogue:\t {len(self.df)}")
        total_utterances = sum([len(item) for item in self.df['turn']])
        logger.info(f"total number of utterance:\t {total_utterances}")

        #  话语内部amr边的类型转id
        edge_types = pd.read_csv("/public/home/zhouxiabing/data/kywang/AMR_MD/data/final/edge_types_amr.csv")
        edge_types = edge_types['edge_types'].to_list()
        self.edge_type2id = {}
        for i, edge_type in enumerate(edge_types):
            self.edge_type2id[edge_type] = i

        self.df['edge_type'] = self.df['edge_type'].apply(
            lambda x: [[self.edge_type2id[item] for item in edges] for edges in x]
        )

        # 话语之间的语篇关系转id
        disc_edge_types = pd.read_csv("/public/home/zhouxiabing/data/kywang/AMR_MD/data/final/edge_types_ddp.csv")
        disc_edge_types = disc_edge_types['edge_types'].to_list()
        self.disc_edge_type2id = {}
        for i, edge_type in enumerate(disc_edge_types):
            self.disc_edge_type2id[edge_type] = i

        self.df['disc_edge_types'] = self.df['disc_edge_types'].apply(
            lambda x: [self.disc_edge_type2id[item] for item in x]
        )

        self.max_seq_len = max_seq_len
        self.max_node_len = max_node_len
        self.tokenizer = tokenizer

        self.id2label = {0: 'Non-malevolent', 1: 'Unconcernedness', 2: 'Detachment', 3: 'Blame', 4: 'Arrogance',
            5: 'Anti-authority', 6: 'Dominance', 7: 'Deceit', 8: 'Negative intergroup attitude (NIA)',
            9: 'Violence', 10: 'Privacy invasion', 11: 'Obscenity', 12: 'Phobia', 13: 'Anger',
            14: 'Jealousy', 15: 'Disgust', 16: 'Self-hurt', 17: 'Immoral and illegal'}
        
        # 1 - ratio
        self.alpha = [0.26619782, 0.98713153, 0.98433837, 0.98209387, 0.96538481, 0.98982493,
                      0.98668263, 0.99361564, 0.98159509, 0.98034815, 0.99211931, 0.97615841,
                      0.99316674, 0.97286648, 0.99186992, 0.97845279, 0.99251833, 0.98563519]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        return self.df.loc[i]
    
    def collate_fn(self, data):
        bsz = len(data)

        max_turns = max([len(dialogue['turn']) for dialogue in data])
        max_num_nodes = max([len(nodes) for dialogue in data for nodes in dialogue['nodes']])
        max_num_edges = max([len(edge) for dialogue in data for edge in dialogue['edge_type']])
        # 话语原始信息
        input_ids = torch.zeros(bsz, max_turns, self.max_seq_len, dtype=torch.int64)
        attention_mask = torch.zeros(bsz, max_turns, self.max_seq_len, dtype=torch.int64)
        labels = torch.full((bsz, max_turns), -1, dtype=torch.int64)
        num_turns = torch.zeros(bsz, dtype=torch.int64)

        # 话语内部句法图
        ## 节点
        num_nodes = torch.zeros(bsz, max_turns, dtype=torch.int64)  # 每个话语的节点数
        node_input_ids = torch.zeros(bsz, max_turns, max_num_nodes, self.max_node_len, dtype=torch.int64)
        node_attention_mask = torch.zeros(bsz, max_turns, max_num_nodes, self.max_node_len, dtype=torch.int64)
        ## 边
        num_edges = torch.zeros(bsz, max_turns, dtype=torch.int64)
        edge_index = torch.full((bsz, max_turns, max_num_edges, 2), -1, dtype=torch.int64)
        edge_types = torch.full((bsz, max_turns, max_num_edges), -1, dtype=torch.int64)

        # 话语之间的语篇关系
        max_disc_num_edges = max([len(dialogue['disc_edge_types']) for dialogue in data])
        num_disc_edges = torch.zeros(bsz, dtype=torch.int64)
        disc_edge_index = torch.full((bsz, max_disc_num_edges, 2), -1, dtype=torch.int64)
        disc_edge_types = torch.full((bsz, max_disc_num_edges), -1, dtype=torch.int64)
        for i in range(bsz):
            # 话语原始信息
            tokenized_text = self.tokenizer(data[i]['clean_text'], max_length=self.max_seq_len, padding='max_length', truncation=True,
                add_special_tokens=True, return_tensors='pt', return_attention_mask=True
            )
            turn = len(data[i]['clean_text'])
            input_ids[i, :turn, :] = tokenized_text['input_ids']
            attention_mask[i, :turn, :] = tokenized_text['attention_mask']
            num_turns[i] = turn
            labels[i, :turn] = torch.tensor(data[i]['label'], dtype=torch.int64)

            # 话语内部句法图
            ## 节点
            # 该对话中，每个话语的节点数
            num_nodes[i, :turn] = torch.tensor([len(utterance_nodes) for utterance_nodes in data[i]['nodes']], dtype=torch.int64)

            # 把该对话的所有话语的所有节点丢到一个列表中，一起tokenizer
            flat_nodes = [node for utterance_nodes in data[i]['nodes'] for node in utterance_nodes]
            tokenized_nodes = self.tokenizer(flat_nodes, max_length=self.max_node_len, padding='max_length', truncation=True,
                add_special_tokens=True, return_tensors='pt', return_attention_mask=True
            )
            # 逐话语填充到目标张量中
            current_idx = 0
            for j in range(turn):
                cnt = num_nodes[i, j]
                node_input_ids[i, j, :cnt, :] = tokenized_nodes['input_ids'][current_idx:current_idx+cnt, :]
                node_attention_mask[i, j, :cnt, :] = tokenized_nodes['attention_mask'][current_idx:current_idx+cnt, :]
                current_idx += cnt
            
            ## 边
            count_utterance_edges = [len(utterance_edge) for utterance_edge in data[i]['edge_type']]
            num_edges[i, :turn] = torch.tensor(count_utterance_edges, dtype=torch.int64)

            for j in range(turn):
                cnt = num_edges[i,j]
                edge_data = torch.tensor(data[i]['edge_index'][j], dtype=torch.int64)
                if edge_data.shape[0] > 0:  # 确保数据不为空
                    edge_index[i, j, :cnt, :] = edge_data
                # edge_index[i, j, :cnt, :] = torch.tensor(data[i]['edge_index'][j], dtype=torch.int64)
                edge_types[i, j, :cnt] = torch.tensor(data[i]['edge_type'][j], dtype=torch.int64)

            # 话语之间的语篇关系
            cur_disc_edges = len(data[i]['disc_edge_types'])
            num_disc_edges[i] = cur_disc_edges
            disc_edge_index[i, :cur_disc_edges, :] = torch.tensor(data[i]['disc_edge_index'], dtype=torch.int64)
            disc_edge_types[i, :cur_disc_edges] = torch.tensor(data[i]['disc_edge_types'], dtype=torch.int64)
        
        """
        num_disc_edges:    [bsz, ]
        disc_edge_index:   [bsz, max_turns, 2]
        disc_edge_types:   [bsz, max_turns]
        """

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'num_turns': num_turns,
            'num_nodes': num_nodes,
            'node_input_ids': node_input_ids,
            'node_attention_mask': node_attention_mask,
            'num_edges': num_edges,
            'edge_index': edge_index,
            'edge_types': edge_types,
            'num_disc_edges': num_disc_edges,
            'disc_edge_index': disc_edge_index,
            'disc_edge_types': disc_edge_types
        }


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("/public/home/zhouxiabing/data/kywang/plms/bert-base-uncased")
    dataset_dir = "/public/home/zhouxiabing/data/kywang/AMR_MD/data/final"
    
    import logging
    logger = logging.getLogger(__name__)
    dataset = AMRDataset(dataset_dir, tokenizer, 128, 12, 'dev', logger)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn)

    for batch in dataloader:
        breakpoint()

