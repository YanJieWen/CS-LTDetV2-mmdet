'''
@File: csnet.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 02, 2024
@HomePage: https://github.com/YanJieWen
'''

from mmdet.registry import MODELS

from glob import glob
import os

import torch
import torch.nn as nn

import math

BatchNorm = nn.BatchNorm2d

class BasicBlock(nn.Module):
    def __init__(self,fan_in,fan_out,stride=1,dilation=1,rfp_fan_in=None):
        super(BasicBlock,self).__init__()
        self. conv1 = nn.Conv2d(fan_in,fan_out,kernel_size=3,stride=stride,padding=dilation,bias=False,dilation=dilation)
        self.bn1 = BatchNorm(fan_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(fan_out,fan_out,kernel_size=3,stride=1,padding=dilation,bias=False,dilation=dilation)
        self.bn2 = BatchNorm(fan_out)
        self.stride = stride
        self.rfp_fanin = rfp_fan_in
        if self.rfp_fanin:
            self.rfp_conv = nn.Conv2d(self.rfp_fanin,fan_out,1,1,bias=False)
    def forward(self,x,residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

    def rfp_forward(self,x,rfp_feat,residual=None): #修改的残差块
        def _inner_forward(x,residual=None):
            if residual is None:
                residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out += residual
            return out

        out = _inner_forward(x,residual)
        if self.rfp_fanin and rfp_feat is not None:
            rpn_feat = self.rfp_conv(rfp_feat)
            out = out+rpn_feat
        out = self.relu(out)

        return out





class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self,fan_in,fan_out,stride=1,dilation=1,rfp_fan_in=None):
        super(Bottleneck,self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = fan_out//expansion
        self.conv1 = nn.Conv2d(fan_in,bottle_planes,kernel_size=1,bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes,fan_out,kernel_size=1,bias=False)
        self.bn3 = BatchNorm(fan_out)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rfp_fanin = rfp_fan_in
        if self.rfp_fanin:
            self.rfp_conv = nn.Conv2d(self.rfp_fanin,fan_out,1,1,bias=False)
    def forward(self,x,residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #中间进行下采样
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

    def rfp_forward(self,x,rfp_feat,residual=None):
        def _inner_forward(x,residual=None):
            if residual is None:
                residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = self.bn3(out)
            out += residual
            return out

        out = _inner_forward(x,residual)
        if self.rfp_fanin and rfp_feat is not None:
            rpn_feat = self.rfp_conv(rfp_feat)
            out = out+rpn_feat
        out = self.relu(out)

        return out

class BottleneckX(nn.Module):
    expansion =2
    cardinality = 32

    def __init__(self,fan_in,fan_out,stride=1,dilation=1,rfp_fan_in=None):
        super(BottleneckX,self).__init__()
        cardinality = self.cardinality
        bottle_planes = fan_out*cardinality//32
        self.conv1  = nn.Conv2d(fan_in,bottle_planes,kernel_size=1,bias=False)
        self.bn1 = BatchNorm(bottle_planes)

        self.conv2 = nn.Conv2d(bottle_planes,bottle_planes,kernel_size=3,stride=stride,padding=dilation,bias=False,
                               dilation=dilation,groups=cardinality)#分组卷积
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, fan_out,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(fan_out)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rfp_fanin = rfp_fan_in
        if self.rfp_fanin:
            self.rfp_conv = nn.Conv2d(self.rfp_fanin, fan_out, 1, 1, bias=False)

    def forward(self,x,residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out
    def rfp_forward(self, x, rfp_feat, residual=None):
        def _inner_forward(x, residual=None):
            if residual is None:
                residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = self.bn3(out)
            out += residual
            return out

        out = _inner_forward(x, residual)
        if self.rfp_fanin and rfp_feat is not None:
            rpn_feat = self.rfp_conv(rfp_feat)
            out = out + rpn_feat
        out = self.relu(out)

        return out


# class SENet(nn.Module):
#     def __init__(self, fan_in, reduce=16) -> None:
#         super(SENet, self).__init__()
#         self.se = nn.AdaptiveAvgPool2d((1, 1))
#         self.ex = nn.Sequential(nn.Linear(fan_in, fan_in // reduce),
#                                 nn.ReLU(inplace=True),
#                                 nn.Linear(fan_in // reduce, fan_in),
#                                 nn.Sigmoid())
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         _se = self.se(x).view(b, c)
#         _ex = self.ex(_se)[:, :, None, None]  # b,c,1,1
#         out = x * _ex
#         # keep_idx = _ex.argsort(dim=-1)[:,-int(self.ratio*c):]#保留前50%大的特征图
#         # out = torch.gather(x,1,keep_idx[:,:,None,None].repeat(1,1,h,w))
#         return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
        # self.se = SENet(in_channels)

    def forward(self, *x):
        children = x
        x = torch.cat(x, 1)
        # x = children[0]
        x = self.conv(x)
        x = self.bn(x)
        if self.residual:
            x += children[0]#如果有残差连接则out_channels等于第一个张量的维度
        x = self.relu(x)

        return x

class Tree(nn.Module):
    def __init__(self,levels,block,in_channels,out_channels,stride=1,level_root=False,root_dim=0,root_kernel_size=1,
                 dilation=1,root_residual=False,rfp_fan_in=None):
        super(Tree,self).__init__()
        if root_dim==0:
            root_dim = 2*out_channels
        if level_root:
            root_dim+=in_channels
        if levels==1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation,rfp_fan_in=rfp_fan_in)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation,rfp_fan_in=None)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,rfp_fan_in=rfp_fan_in)#类套类
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual,rfp_fan_in=None)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root#是否跨阶段连接，当为False时跨阶段连接被取消
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:#分辨率对齐
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:#通道对齐
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )
    def forward(self, x, residual=None, children=None):#迭代聚合
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x
    def rfp_forward(self,x, residual=None, children=None,rfp_feat=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1.rfp_forward(x, rfp_feat=rfp_feat,residual=residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@MODELS.register_module()
class CSNet(nn.Module):
    arch_settings = {
        34: ([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], BasicBlock,False),
        46: ([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], Bottleneck,False),
        60: ([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], Bottleneck,False),
        102: ([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], Bottleneck,True),
        169: ([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],Bottleneck,True)
    }
    def __init__(self,
                 depth: int,
                 out_indices:tuple=(2,3,4,5),
                 level_root:bool=True,
                 output_img:bool=False,
                 rfp_inplanes:int=None):
        super(CSNet,self).__init__()
        assert depth in CSNet.arch_settings.keys(), f'{int(depth)} must in the [34,46,60,102,169]'
        levels,channels,block,residual_root = CSNet.arch_settings[depth]
        try:
            pretrain = glob(os.path.join('./pretrain',f'dla{int(depth)}*'))[0]
        except:
            raise ValueError(f'{depth}-model is not found')

        ckpt = torch.load(pretrain, map_location='cpu')

        self.num_levels = len(levels)
        self.out_indices = out_indices
        # if self.rfp_fanin is not None:
        #     self.maxpool = nn.MaxPool2d(2,2)

        self.base_layer = nn.Sequential(nn.Conv2d(3,channels[0],kernel_size=7,stride=1,padding=3,bias=False),
                                        BatchNorm(channels[0]),
                                        nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)#2
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,root_residual=residual_root,rfp_fan_in=rfp_inplanes)#4
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=level_root, root_residual=residual_root,rfp_fan_in=rfp_inplanes)#8
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=level_root, root_residual=residual_root,rfp_fan_in=rfp_inplanes)#16
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=level_root, root_residual=residual_root,rfp_fan_in=rfp_inplanes)#32
        self.apply(self.init_weights_)
        #加载预训练权重
        mis_keys,_ = self.load_state_dict(ckpt,strict=False)
        # #冻结前置层
        # self._freeze_stages()
        # #缺少的键需要训练
        # for name,m in self.named_parameters():
        #     if name in mis_keys:
        #         m.requires_grad = True
        self.output_img = output_img
    def init_weights_(self,m):
        if isinstance(m,nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, BatchNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)
    #
    # def _freeze_stages(self):
    #     for m in [self.base_layer,self.level0]:
    #         for parm in m.parameters():
    #             parm.requires_grad = False
    #     for i in range(1,self.frozen_stages+1):
    #         m = getattr(self,f'level{i}')
    #         m.eval()
    #         for parm in m.parameters():
    #             parm.requires_grad = False

    def forward(self,x):
        outs = []
        y = self.base_layer(x)
        for i in range(self.num_levels):
            y = getattr(self,'level{}'.format(int(i)))(y)
            if self.out_indices is not None:
                if i in self.out_indices:
                    outs.append(y)
            else:
                outs = y
        if self.output_img:
            outs.insert(0,x)
        return tuple(outs)

    def rfp_forward(self,x,rpn_feats):
        x = self.base_layer(x)
        outs = []
        num_fea = 0
        for i in range(self.num_levels):
            level_layer = getattr(self, 'level{}'.format(int(i)))
            if i>1:
                rfp_feat = rpn_feats[num_fea]
                x = level_layer.rfp_forward(x,rfp_feat=rfp_feat)
                num_fea+=1
            else:
                x = level_layer(x)
            if i in self.out_indices:
                outs.append(x)
            # print(f'{int(i)}stage--{x.shape}')
        return tuple(outs)










# if __name__ == '__main__':
#     from glob import glob
#     import os
#     pretrain = 'dla34'
#     pretrain = glob(os.path.join('./pretrain',f'{pretrain}*'))[0]
#     ckpt = torch.load(pretrain,map_location='cpu')
#     x = torch.rand((2,3,224,224))
#     rpn_feats = [torch.rand((2,256,56,56)),torch.rand((2,256,28,28)),torch.rand((2,256,14,14)),
#                  torch.rand((2,256,7,7))]
#     model = CSNet(depth=34,out_indices=(2,3,4,5),rfp_inplanes=256)
#     mis,exp = model.load_state_dict(ckpt,strict=False)#ckpt缺少的键盘，exp为ckpt多出的键
#     # for name,m in model.named_parameters():
#     #     if name in mis:
#     #         m.requires_grad = True
#     # print(mis,exp)
#     [print(y.shape) for y in model.rfp_forward(x,rpn_feats)]
#     # print(model)