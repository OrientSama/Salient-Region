import os
import json
import numpy as np


def main():
    json_path = './class_indices.json'
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            class_indices = json.load(f)
            # 调整键值对 顺序
            class_indices = dict((k, int(v)) for v, k in class_indices.items())
    root_path = r"D:\project\Data\trainSplit-1024\labelTxt"
    ann_list = [os.path.join(root_path, i) for i in os.listdir(root_path)]
    class_sum = np.zeros(16)
    for ann in ann_list:
        ann_class = np.zeros(16)
        with open(ann, 'r') as f:
            data = f.readlines()
            if len(data) != 0:
                tmp = set()
                for line in data:
                    tmp.add(line.split(' ')[-2])
                tmp_l = list(tmp)
                for t in tmp_l:
                    ann_class[int(class_indices[t])] = 1.0
            else:
                ann_class[int(class_indices['background'])] = 1.0
            class_sum += ann_class
    cla_name = list(class_indices.keys())
    # for cla in class_sum:
    with open('./analyse.txt', 'w') as ana:
        for i in range(16):
            ana.write('{:20} : {:6}  {:.3f}%\n'.format(cla_name[i], class_sum[i], class_sum[i] * 100 / len(ann_list)))


if __name__ == "__main__":
    main()
