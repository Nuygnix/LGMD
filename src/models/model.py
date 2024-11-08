import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig
from torch_geometric.nn import RGCNConv, RGATConv

import sys
sys.path.append("..")
from data_modules.dataset import AMRDataset
from torch.utils.data import Dataset, DataLoader
import argparse
from models.pooling import create_pool_layer

# BASE_MODEL_MAP


# Local semantics
class AMRModel(nn.Module):
    def __init__(self, args, num_labels):
        super().__init__()
        self.args = args
        self.bert_config = BertConfig.from_pretrained(args.plm_path)
        self.utterance_encoder = BertModel.from_pretrained(args.plm_path)
        self.utterance_pool_layer = create_pool_layer("cls")

        in_dim = self.bert_config.hidden_size

        if args.with_inner_syntax:
            # 可以与utterance encoder共享参数
            self.node_encoder = BertModel.from_pretrained(args.plm_path)
            # TODO: 加个Linear
            self.node_init_pool_layer = create_pool_layer("attention", self.bert_config.hidden_size)
            self.conv1 = RGATConv(in_channels=self.bert_config.hidden_size,
                                out_channels=args.node_hidden_size1,
                                num_relations=args.num_relations,
                                num_bases=30, dropout=0.05)
            self.conv2 = RGATConv(in_channels=args.node_hidden_size1,
                                out_channels=args.node_hidden_size2,
                                num_relations=args.num_relations,
                                num_bases=30, dropout=0.05)

            # # TODO: 加个Linear
            # self.node_fc1 = nn.Linear(self.bert_config.hidden_size, args.node_hidden_size)
            # # TODO: 加个Dropout
            # self.node_init_pool_layer = create_pool_layer("attention", self.bert_config.hidden_size)
            # self.conv_layers = []
            # for _ in range(args.num_node_gnn_layer):
            #     self.conv_layers.append(RGCNConv(in_channels=args.node_hidden_size,
            #                     out_channels=args.node_hidden_size,
            #                     num_relations=args.num_relations,
            #                     num_bases=30))
            
            # TODO: Attention Pooling / get root features
            in_dim += args.node_hidden_size
        
        # classifier
        layers = [nn.Linear(in_dim, args.clf_hidden_size), nn.ReLU()]
        for _ in range(args.num_clf_layers - 1):
            layers += [nn.Linear(args.clf_hidden_size, args.clf_hidden_size), nn.ReLU()]
        layers += [nn.Linear(args.clf_hidden_size, num_labels)]

        self.clf = nn.Sequential(*layers)

    
    def forward(self, **kwargs):
        """
        batch个对话
        input_ids: [bsz, max_turn, max_seq_len]
        attention_mask: [bsz, max_turn, seq_len]
        labels: [bsz, max_turns] 填充的-1
        num_turns: [bsz, ]
        """
        device = kwargs['input_ids'].device
        bsz, max_turns, max_seq_len = kwargs['input_ids'].shape
        total_valid_utterances = kwargs['num_turns'].sum().item()  # 有效话语的总数
        # 展平
        input_ids_flat = kwargs['input_ids'].view(-1, max_seq_len)
        attention_mask_flat = kwargs['attention_mask'].view(-1, max_seq_len) 

        # 有效样本掩码
        valid_mask = torch.zeros(bsz * max_turns, dtype=torch.bool).to(device)
        for i in range(bsz):
            start_idx = i * max_turns
            valid_mask[start_idx: start_idx + kwargs['num_turns'][i].item()] = True

        valid_input_ids = input_ids_flat[valid_mask]  # [有效样本数, max_seq_len]
        valid_attention_mask = attention_mask_flat[valid_mask]  # [有效样本数, max_seq_len]

        outputs = self.utterance_encoder(valid_input_ids, valid_attention_mask)
        origin_features = self.utterance_pool_layer(outputs.last_hidden_state) # [有效样本数, hidden_size]

        final_features = [origin_features]

        if self.args.with_inner_syntax:
            all_node_input_ids, all_attention_mask, all_edge_index, all_edge_type = self.batch_graphify_inner(
                kwargs['num_nodes'], kwargs['node_input_ids'], kwargs['node_attention_mask'],
                kwargs['num_edges'], kwargs['edge_index'], kwargs['edge_types'])
            
            outputs = self.node_encoder(all_node_input_ids, all_attention_mask)
            node_init_features = self.node_init_pool_layer(outputs.last_hidden_state, all_attention_mask)    # [total_num_nodes, hidden_size]
            
            node_features = self.conv1(node_init_features, all_edge_index, all_edge_type)
            node_features = self.conv2(node_features, all_edge_index, all_edge_type)

            # node_features = self.node_fc1(node_init_features)
            # for i in range(self.args.num_node_gnn_layer):
            #     node_features = self.conv_layers[i](node_features, all_edge_index, all_edge_type)

            # TODO: 过GNN之前和之后的可以一起用，残差连接 / concat
            # node_features = torch.cat([node_features, node_init_features], dim=1)
            
            # 计算每个话语的起始和结束位置
            node_starts = kwargs['num_nodes'].view(-1).cumsum(dim=0) - kwargs['num_nodes'].view(-1)  # [bsz * max_turns] 每个话语的全局起始索引
            node_ends = node_starts + kwargs['num_nodes'].view(-1)
            
            inner_features = torch.zeros(total_valid_utterances, node_features.shape[1]).to(device)

            idx = 0
            for i in range(bsz):
                for j in range(kwargs['num_turns'][i]):
                    start = node_starts[i * max_turns + j].item()
                    end = node_ends[i * max_turns + j].item()
                        # TODO: 可以改成自注意力聚合
                    utterance_feat = node_features[start: end].mean(dim=0)
                    inner_features[idx] = utterance_feat
                    idx += 1
            final_features.append(inner_features)


        final_features = torch.cat(final_features, dim=1)
        logits = self.clf(final_features)# [有效样本数, num_classes]
        loss = None
        if 'labels' in kwargs:
            loss_fn = nn.CrossEntropyLoss()
            labels_flat = kwargs['labels'].view(-1)
            valid_labels = labels_flat[valid_mask]  # [有效样本数]
            loss = loss_fn(logits, valid_labels)
        return (logits, valid_labels), loss
        
        # # edge_index [2, num_edges]
        # # edge_type [num_edges]

    def batch_graphify_inner(self, num_nodes, node_input_ids, node_attention_mask,
                             num_edges, edge_index, edge_types):
        """
        num_nodes: [bsz, max_turn, ]
        node_input_ids: [bsz, max_turn, max_num_nodes, max_node_len]
        node_attention_mask: [bsz, max_turn, max_num_nodes, max_node_len]

        num_edges: [bsz, max_turn, ]
        edge_index: [bsz, max_turn, max_num_edges, 2]
        edge_types: [bsz, max_turn, max_num_edges]
        """
        # 把batch内所有话语内部节点图，构成一张大图

        bsz, max_turns = num_nodes.shape
        all_node_input_ids = []
        all_attention_mask = []
        all_edge_index = []
        all_edge_type = []
        node_offset = 0
        for b in range(bsz):
            for t in range(max_turns):
                # 提取节点和attention_mask
                curr_num_nodes = num_nodes[b, t].item()
                nodes = node_input_ids[b, t, :curr_num_nodes]  # [curr_num_nodes, max_node_len]
                attention_mask = node_attention_mask[b, t, :curr_num_nodes]  # [curr_num_nodes, max_node_len]

                # 将节点数据添加到大图的节点列表中
                all_node_input_ids.append(nodes)
                all_attention_mask.append(attention_mask)

                # 边的索引和类型
                curr_num_edges = num_edges[b, t].item()
                edges = edge_index[b, t, :curr_num_edges]  # [curr_num_edges, 2]
                types = edge_types[b, t, :curr_num_edges]  # [curr_num_edges]

                # 更新边索引以适应全局节点编号
                edges = edges + node_offset  # 偏移量使得所有话语节点编号不冲突
                all_edge_index.append(edges)
                all_edge_type.append(types)

                # 更新偏移量
                node_offset += curr_num_nodes

        # 合并所有节点、attention mask、边索引和边类型
        all_node_input_ids = torch.cat(all_node_input_ids, dim=0)  # [total_num_nodes, max_node_len]
        all_attention_mask = torch.cat(all_attention_mask, dim=0)  # [total_num_nodes, max_node_len]
        all_edge_index = torch.cat(all_edge_index, dim=0).t().contiguous()  # [2, total_num_edges]
        all_edge_type = torch.cat(all_edge_type, dim=0)  # [total_num_edges]
        return all_node_input_ids, all_attention_mask, all_edge_index, all_edge_type

    def batch_graphify(self, ):
        # 把batch内所有话语，构成一张大图
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The arguments of Conversation")

    parser.add_argument("--plm_path", type=str, default="/public/home/zhouxiabing/data/kywang/plms/bert-base-uncased")
    parser.add_argument("--node_hidden_size1", type=int, default=200)
    parser.add_argument("--node_hidden_size2", type=int, default=200)
    parser.add_argument("--clf_hidden_size", type=int, default=100)
    parser.add_argument("--num_clf_layers", type=int, default=1)
    parser.add_argument("--num_relations", type=int, default=116)

    parser.add_argument("--with_inner_syntax", action='store_true')


    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.plm_path)
    dataset_path = "/public/home/zhouxiabing/data/kywang/AMR_MD/data/final/test_mdrdc.jsonl"
    dataset = AMRDataset(dataset_path, tokenizer, 256, 12)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

    device = 'cuda'
    model = AMRModel(args, 18)
    model.to(device)

    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        logits, loss = model(**batch)




