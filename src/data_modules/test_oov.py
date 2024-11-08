
from transformers import BartTokenizer, BartConfig, BertConfig, BertTokenizer

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    df = pd.read_json("/public/home/zhouxiabing/data/kywang/AMR_MD/data/amr_data.jsonl", lines=True)

    tokenizer = BertTokenizer.from_pretrained("/public/home/zhouxiabing/data/kywang/plms/bert-base-uncased")
    cnt = 0
    unk_id = tokenizer(tokenizer.unk_token, add_special_tokens=False)['input_ids'][0]
    for nodes in tqdm(df['nodes']):
        for node in nodes.values():
            tokenized_node = tokenizer(node, add_special_tokens=False)['input_ids']
            for i in tokenized_node:
                if i == unk_id:
                    cnt += 1

    print(cnt)  # 0


