import torch
import torch.nn as nn


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
