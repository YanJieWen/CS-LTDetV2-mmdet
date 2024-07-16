'''
@File: get_pr.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 11, 2024
@HomePage: https://github.com/YanJieWen
'''


import os
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pandas as pd
from glob import glob

gt_path = '../data/crash2024/annotations/test.json'
pred_path = './double_head_rcnn.json'
M = 2
thr = -1

coco = COCO(gt_path)
coco_dt =coco.loadRes(pred_path)
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

p_value = coco_eval.eval['precision']
recall = np.mat(np.arange(0.0, 1.01, 0.01)).T
if thr==-1:
    map_all_pr = np.mean(p_value[:, :, :, 0, M], axis=0)
else:
    T = int((thr - 0.5) / 0.05)
    map_all_pr = p_value[T, :, :, 0, M]

data = np.hstack((np.hstack((recall, map_all_pr)),
                  np.mat(np.mean(map_all_pr, axis=1)).T))
df = pd.DataFrame(data)
out_name = os.path.basename(pred_path).split('.')[0]
save_path = f'./{out_name}.xlsx'
df.to_excel(save_path, index=False)




