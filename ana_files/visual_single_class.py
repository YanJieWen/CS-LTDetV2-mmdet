'''
@File: visual_single_class.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 11, 2024
@HomePage: https://github.com/YanJieWen
'''

import sys
import numpy as np
import os
import os.path as osp
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL.ImageDraw as ImageDraw
import time
import pickle

from pycocotools.coco import COCO


def visual_iou(gt,pred):
    gt_area = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])
    pred_area = (pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])
    lt = np.maximum(gt[:,None,:2],pred[:,:2])
    rb = np.minimum(gt[:,None,2:],pred[:,2:])
    wh = (rb-lt).clip(0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (gt_area[:, None] + pred_area - inter)
    return iou

def draw_agnostic(img,boxes,color=(239, 35, 60),line_thicknes=8):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        left,top,right,bottom = box
        draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thicknes, fill=color)
    return img

gt_path ='../data/crash2024/annotations/test.json'
pred_path = './double_head_rcnn.json' #按照图片加载pkl文件
assert osp.isfile(pred_path),f'the {pred_path} is not exists'
model_name = osp.basename(pred_path).split('.')[0]
output_root = f'./{model_name}'
if not osp.exists(output_root):
    os.makedirs(output_root)
image_root = '../data/crash2024/test'
iou_theresh = 0.5

gt_coco = COCO(gt_path)
pred_coco = gt_coco.loadRes(pred_path)
coco_classes = dict([(str(v['id']),v['name']) for k,v in gt_coco.cats.items()])
img_ids = [k for k,_ in gt_coco.imgs.items()]
pbar = tqdm(img_ids,desc='Visualization images...',file=sys.stdout)
for id in pbar:
    img_path = gt_coco.imgs[id]['file_name']
    img_path = osp.join(image_root,img_path)
    gt_ann_ids = gt_coco.getAnnIds(id)
    tgts = gt_coco.loadAnns(gt_ann_ids)
    tgt_locs = np.concatenate([[v for k, v in label.items() if k == 'bbox'] for label in tgts], axis=0)
    tgt_cats = np.concatenate([[v for k, v in label.items() if k == 'category_id'] for label in tgts], axis=0).reshape(
        -1, 1)
    tgts = np.concatenate((tgt_locs, tgt_cats), axis=1)
    tgts[:, 2] = tgts[:, 0] + tgts[:, 2]
    tgts[:, 3] = tgts[:, 1] + tgts[:, 3]

    pred_ids = pred_coco.getAnnIds(id)
    preds = pred_coco.loadAnns(pred_ids)
    try:
        pred_locs = np.concatenate([[v for k, v in x.items() if k == 'bbox']  for x in preds])
        pred_cats = np.concatenate([[v for k, v in label.items() if k == 'category_id'] for label in preds], axis=0).reshape(
            -1, 1)
    except:
        pred_locs = np.full((0,4),0)
        pred_cats = np.full((0,1),0)
    preds = np.concatenate((pred_locs, pred_cats), axis=1)
    preds[:, 2] = preds[:, 0] + preds[:, 2]
    preds[:, 3] = preds[:, 1] + preds[:, 3]

    img = cv2.imread(img_path)[:, :, ::-1]
    img = Image.fromarray(img)
    tgt_ps = tgts[tgts[:, -1] == 1, :4]
    pred_ps = preds[preds[:, 4] == 1, :4]
    if len(pred_ps) != 0:
        # 计算IOU损失
        tgt_pred_cost = visual_iou(tgt_ps, pred_ps)
    else:
        continue

    tp_pred_ids = np.argmax(tgt_pred_cost, axis=1)
    tp_gt_ids = np.argmax(tgt_pred_cost, axis=0)
    gt_matches = set([(i, idx) for i, idx in enumerate(tp_pred_ids)])
    pred_matches = set([(idx, i) for i, idx in enumerate(tp_gt_ids)])
    mathes_tuple = list(gt_matches.intersection(pred_matches))
    matches_ids = np.array([[idx[0], idx[1]] for idx in mathes_tuple])
    tp_dual_ids = np.where(tgt_pred_cost[matches_ids[:, 0], matches_ids[:, 1]] > iou_theresh)[0]  # 高于0.5的pred索引
    if len(tp_dual_ids) != 0:
        _tp_ids = matches_ids[tp_dual_ids][:, 1]
        _fp_ids = list(set(list(np.arange(pred_ps.shape[0]))) - set(_tp_ids))  # 误检的pred索引
        _fn_ids = list(set(list(np.arange(tgt_ps.shape[0]))) - set(matches_ids[tp_dual_ids][:, 0]))  # 漏检的索引
        tp_boxes = pred_ps[_tp_ids]
        fp_boxes = pred_ps[_fp_ids]
        fn_boxes = tgt_ps[_fn_ids]
        img = draw_agnostic(img, tp_boxes, color=(239, 35, 60), line_thicknes=3)  # 红色
        img = draw_agnostic(img, fp_boxes, color=(58, 134, 255), line_thicknes=3)  # 误检-蓝色
        img = draw_agnostic(img, fn_boxes, color=(42, 157, 143), line_thicknes=3)  # 漏检-绿色
    else:
        pbar.desc = f'{str(int(id)).zfill(6)} no any point class target'
        continue
    img = np.array(img)[:, :, ::-1]
    save_path = osp.join(output_root, f'{str(int(id)).zfill(6)}.jpg')
    cv2.imwrite(save_path, img)
    time.sleep(1)