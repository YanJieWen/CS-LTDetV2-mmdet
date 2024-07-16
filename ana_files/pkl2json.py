'''
@File: pkl2json.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 10, 2024
@HomePage: https://github.com/YanJieWen
'''

import os
import sys

import pickle
import json
from glob import glob
from tqdm import tqdm

def parse_single_pkl(data_path):
    out_name = os.path.basename(data_path).split('.')[0]
    out_path = f'{out_name}.json'
    with open(data_path, 'rb') as r:
        data = pickle.load(r)
    results = []
    for pred in data:
        pred_info = pred['pred_instances']
        image_id = int(pred['img_id'])
        for s, c, b in zip(pred_info['scores'].numpy(), pred_info['labels'].numpy(), pred_info['bboxes'].numpy()):
            category_id = int(c + 1)
            bbox = [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
            score = float(s)
            results.append({"image_id": image_id, 'category_id': category_id,
                            'bbox': bbox, 'score': score})
    x = json.dumps(results)
    with open(out_path, 'w') as w:
        w.write(x)
    w.close()

if __name__ == '__main__':
    data_root = './*.pkl'
    data_paths = glob(data_root)
    # print(data_paths)
    pbar = tqdm(data_paths,file=sys.stdout)
    for data in pbar:
        parse_single_pkl(data)