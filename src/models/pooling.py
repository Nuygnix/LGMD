import torch
import torch.nn as nn
import torch.nn.functional as F


def create_pool_layer(method, in_dim=None):
    assert method in ['cls', 'attention', 'mean', 'min', 'max'], "only support cls/attention/mean/min/max!"
    if method == "cls":
        return CLSPooling()
    elif method == 'mean':
        return MeanPooling()
    elif method == 'min':
        return MinPooling()
    elif method == 'max':
        return MaxPooling()
    else:
        return AttentionPooling(in_dim)



class CLSPooling(nn.Module):
    def __init__(self):
        super(CLSPooling, self).__init__()
    
    def forward(self, last_hidden_state, attention_mask=None):
        return last_hidden_state[:, 0]


class NodeAttentionPooling(nn.Module):
    def __init__(self, in_dim):
        """
        初始化注意力池化层
        :param in_dim: 输入节点表征的维度
        """
        super(NodeAttentionPooling, self).__init__()
        # self.attention_weights = nn.Linear(in_dim, 1)  # 学习注意力权重的线性变换
        # TODO: 两层Linear
        self.attention_weights = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, node_features, mask=None):
        """
        前向传播
        :param node_features: (N, D) 张量，表示图中 N 个节点的 D 维表征
        :param mask: (N,) 可选的掩码，用于屏蔽某些无效节点
        :return: 话语级别的全局表征 (1, D)
        """
        # 计算注意力得分 (N, 1)
        scores = self.attention_weights(node_features)  # (N, 1)
        scores = F.leaky_relu(scores)  # 激活函数（可选）

        # 归一化权重，使用 softmax
        attn_weights = F.softmax(scores, dim=0)  # (N, 1)

        # 如果有掩码，屏蔽无效节点的权重
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(-1)

        # 根据权重对节点表征加权汇聚
        graph_representation = torch.sum(attn_weights * node_features, dim=0)  # (D,)

        return graph_representation


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    

class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
        embeddings = last_hidden_state.masked_fill(~input_mask_expanded, float('inf'))
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings
