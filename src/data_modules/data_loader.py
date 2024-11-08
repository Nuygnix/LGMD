import sys
sys.path.append("..")

from data_modules.dataset import AMRDataset
from torch.utils.data import DataLoader, Subset

def get_dataloader(tokenizer, dataset_dir, split,
                   batch_size, max_seq_len, max_node_len, logger):
    """
    dataset_dir     处理好的数据集文件目录
    split           train, dev, test的一个
    max_seq_len     每个话语的最大长度
    max_node_len    每个amr节点token的最大长度
    """
    # dataset_path, tokenizer, max_seq_len, max_node_len)
    dataset = AMRDataset(dataset_dir, tokenizer, max_seq_len, max_node_len, split, logger)
    
    # return DataLoader(Subset(dataset, range(200)), batch_size=batch_size, shuffle=split == 'train')
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=split == 'train', collate_fn=dataset.collate_fn)