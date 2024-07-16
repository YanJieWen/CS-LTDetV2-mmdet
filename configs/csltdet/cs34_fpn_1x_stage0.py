'''
@File: cs34_fpn_1x_stage0.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 02, 2024
@HomePage: https://github.com/YanJieWen
'''


_base_ = './cs34_fpn_1x.py'
model = dict(
    backbone=dict(
        type='CSNet',
        depth=34,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1),
    neck=dict(
        type='FPN',
        in_channels=[16,32,64,128], #1,2,3,4
        out_channels=256,
        num_outs=5)
)