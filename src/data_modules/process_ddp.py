"""
We use JSON format for data. The data file should contain an array consisting of examples represented as objects. The format for each example object looks like:
{
    "ids": "xxx"
    // a list of EDUs in the dialogue
    "edus": [ 
        {
            "text": "text of edu 1",
            "speaker": "speaker of edu 1"
        },    
        // ...
    ],
    // a list of relations
    "relations": [
        {
            "x": 0,
            "y": 1,
            "type": "type 1"
        },
        // ...
    ]
}

"""
import json
import pandas as pd
DATA_DIR = "/public/home/zhouxiabing/data/kywang/AMR_MD/data/"


def step1():
    data_path = "/public/home/zhouxiabing/data/kywang/AMR_MD/data/final/dev_mdrdc.jsonl"
    output_path = "dev.json"
    res = []
    with open(data_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            edus = []
            i = 0
            for text in data['clean_text']:
                if i % 2 == 0:
                    speaker = 'A'
                else:
                    speaker = 'B'
                i += 1
                edus.append({'text': text, 'speaker': speaker})
            res.append({"id": data['number'], "edus": edus, "relations": []})
    
    with open(output_path, "w") as f:
        f.write(json.dumps(res))



def step2():
    split = "dev"
    data = json.load(open(f'{DATA_DIR}/intermediate/{split}_ddp.json', 'r'))
    df = pd.read_json(f'{DATA_DIR}/final/{split}_mdrdc.jsonl', lines=True)
    relations = [item['relations'] for item in data]
    edge_index = []
    edge_type = []
    num_edges = []
    for r in relations:
        tmp1 = []
        tmp2 = []
        num_edges.append(len(r))
        for item in r:
            tmp1.append([item['x'], item['y']])
            tmp2.append(item['type'])
        edge_index.append(tmp1)
        edge_type.append(tmp2)

    df['disc_edge_index'] = edge_index
    df['disc_edge_types'] = edge_type
    df['num_disc_edges'] = num_edges

    df.to_json(f"{DATA_DIR}/final/{split}_mdrdc.jsonl", orient="records", lines=True)


if __name__ == '__main__':
    # step1() # 处理完交给DDP模型解析
    step2()


