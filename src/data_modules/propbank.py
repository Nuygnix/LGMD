import re
import os
from lxml import etree as ET

def get_definition(node_name):
    word = re.sub(r'-\d+$', '', node_name)
    filepath = f"/public/home/zhouxiabing/data/kywang/AMR_MD/propbank-frames-main/frames/{word}.xml"
    if not os.path.exists(filepath):
        return None
    tree = ET.parse(filepath)
    root = tree.getroot()
    # 遍历所有 roleset 元素
    target_id = re.sub(r'-(\d+)$', r'.\1', node_name)
    for roleset in root.iter("roleset"):
        if roleset.get("id") == target_id:
            return roleset.get("name")
    return None


def trans_node(node_name):
    # 定义正则模式
    pattern = r"\b([a-zA-Z]+)-[0-9]+\b"

    if re.fullmatch(pattern, node_name):
        definition = get_definition(node_name)
        # 没有对应定义，直接用去掉-数字的
        if definition is None:
            return re.sub(r'-\d+$', '', node_name)
        else:
            # return definition
            return re.sub(r'-\d+$', '', node_name) + ":" + definition
    else:
        return node_name


if __name__ == '__main__':
    with open("/public/home/zhouxiabing/data/kywang/amr_md/data/node_set.txt", "r") as f, open("/public/home/zhouxiabing/data/kywang/amr_md/data/output_define.txt", "w") as f2:
        while True:
            line = f.readline().strip()
            if not line:
                break

            # f2.write(trans_node(line) + "\n")
            
            if bool(re.search(r'-\d+$', line)):
                definition = get_definition(line)
                if definition is None:
                    f2.write(f"Error: Can't find {line}\n")
                else:
                    f2.write(f"Right: {line} ----->  {definition}\n")
            else:
                f2.write(f"Origin: {line}\n")