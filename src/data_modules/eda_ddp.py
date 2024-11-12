import json
from collections import Counter, defaultdict

DATA_DIR = "/public/home/zhouxiabing/data/kywang/AMR_MD/data"

# 加载数据
with open(f'{DATA_DIR}/intermediate/train_ddp.json', 'r') as f:
    data = json.load(f)

# 初始化计数器
edge_count_dist = []
degree_dist = defaultdict(int)
edge_type_dist = Counter()
dialogue_edu_counts = []  # 存储每个对话的 EDU 数量
total_edges = 0           # 统计所有对话中的总边数

# 遍历每个对话
for dialogue in data:
    # 获取 EDU 节点数
    edus = dialogue['edus']
    num_nodes = len(edus)
    dialogue_edu_counts.append(num_nodes)  # 添加话语数到列表
    
    # 获取关系边数
    relations = dialogue['relations']
    edge_count = len(relations)
    edge_count_dist.append(edge_count)
    total_edges += edge_count  # 累计总边数
    
    # 初始化每个节点的度数
    node_degrees = [0] * num_nodes
    
    # 遍历每条边
    for relation in relations:
        x, y = relation['x'], relation['y']
        edge_type = relation['type']
        
        # 记录边类型分布
        edge_type_dist[edge_type] += 1
        
        # 更新出度和入度
        node_degrees[x] += 1
        node_degrees[y] += 1

    # 将每个节点的度数添加到总体分布
    for degree in node_degrees:
        degree_dist[degree] += 1

# 计算平均边数
average_edge_count = total_edges / len(data)

# 输出统计结果
print("每个对话的话语数分布：", Counter(dialogue_edu_counts))
print("每个对话的边数分布：", Counter(edge_count_dist))
print("每个话语的度数分布：", dict(degree_dist))
print("边类型分布：", dict(edge_type_dist))
print("每个对话的平均边数：", average_edge_count)

import pandas as pd
df = pd.DataFrame(list(dict(edge_type_dist).items()), columns=['edge_types', 'count'])
df.to_csv(f"{DATA_DIR}/final/edge_types_ddp.csv", index=False)