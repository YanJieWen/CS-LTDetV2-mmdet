'''
@File: my_ana.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 10, 2024
@HomePage: https://github.com/YanJieWen
'''

import os
from mmdet.apis import init_detector,inference_detector
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as ts
import numpy as np
import mmcv
import cv2

from myvisual_utils import draw_objs


from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose

class ActivationAndGradients:
    def __init__(self,model,target_layers,reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for tgt_ls in target_layers:
            self.handles.append(tgt_ls.register_forward_hook(hook=self.save_activation))
            self.handles.append(tgt_ls.register_backward_hook(hook=self.save_gradient))
    def save_activation(self,module,input,output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
    def save_gradient(self,module,grad_input,grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()]+self.gradients
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model.test_step(x)[0]
    def release(self):
        for handle in self.handles:
            handle.remove()


class Grad_cam():
    def __init__(self,img_file,configs_files,ckpt_file,if_outfea):
        self.model = init_detector(config=configs_files, checkpoint=ckpt_file,
                                   device='cuda:0')
        self.model.eval()
        self.outfea = if_outfea
        self.img_file = img_file
        self.img = Image.open(img_file).convert('RGB')
        self.activations_and_grads = ActivationAndGradients(self.model,
                                                          [self.model.backbone.level5],reshape_transform=None)
        self.data_warp = None
    @staticmethod
    def get_2d_projection(activation_batch):
        # TBD: use pytorch batch svd implementation
        activation_batch[np.isnan(activation_batch)] = 0
        projections = []
        for activations in activation_batch:
            reshaped_activations = (activations).reshape(
                activations.shape[0], -1).transpose()
            # Centering before the SVD seems to be important here,
            # Otherwise the image returned is negative
            reshaped_activations = reshaped_activations - \
                                   reshaped_activations.mean(axis=0)
            U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
            projection = reshaped_activations @ VT[0, :]
            projection = projection.reshape(activations.shape[1:])
            projections.append(projection)
        return np.float32(projections)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result
    def get_pred(self,res):
        # plt.axis('off')
        labels = res.pred_instances.labels.detach().cpu().numpy()
        scores = res.pred_instances.scores.detach().cpu().numpy()
        bboxes = res.pred_instances.bboxes.detach().cpu().numpy()
        _img = draw_objs(self.img, bboxes, labels, scores, line_thickness=12, font_size=48)
        return _img
    @staticmethod
    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join('./demo', fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    def compute_cam_per_layer(self,input_tensor):
        act_list = [x.cpu().data.numpy() for x in self.activations_and_grads.activations]
        grad_list = [x.cpu().data.numpy() for x in self.activations_and_grads.gradients]
        img_shapes = input_tensor['inputs'][0].shape
        width, height = img_shapes[-1], img_shapes[-2]
        tgt_size = (width, height)
        cam_per_tgt_layer = []
        for layer in act_list:
            cam = self.get_2d_projection(layer)
            cam[cam < 0] = 0
            scaled = self.scale_cam_image(cam, tgt_size)
            cam_per_tgt_layer.append(scaled[:, None, :])
        return cam_per_tgt_layer

    def aggregate_multi_layers(self,cam_per_layer):
        cam_per_tgt_layer = np.concatenate(cam_per_layer, axis=1)
        cam_per_tgt_layer = np.maximum(cam_per_tgt_layer, 0)
        result = np.mean(cam_per_tgt_layer, axis=1)
        result = self.scale_cam_image(result)
        return result

    def __call__(self):
        imgs = [self.img_file]
        cfg = self.model.cfg
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline = Compose(test_pipeline)
        # self.model.eval()
        for i, img in enumerate(imgs):
            if isinstance(img, np.ndarray):
                data_ = dict(img=img, img_id=0)
            else:
                data_ = dict(img_path=img, img_id=0)
        data_ = test_pipeline(data_)
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        self.data_warp = data_
        res = self.activations_and_grads(data_)
        if self.outfea:
            img = self.get_pred(res)
            img.save('./demo/pred.jpg')
        cam_per_layer = self.compute_cam_per_layer(data_)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_img(mask,input_tensor):
    '''

    :param mask: [1,h,w]
    :param input_tensor:dict-->rgb
    :return: [h,w,3]
    '''
    cam = mask[0, :]
    show_img = input_tensor['inputs'][0].numpy().astype(dtype=np.float32) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + show_img.transpose(1, 2, 0)
    cam = cam / np.max(cam)
    show_cam = np.uint8(255 * cam)
    return show_cam

if __name__ == '__main__':
    img_file = './data/crash2024/test/006342.jpg'
    configs_files = './configs/csltdet/cs60_rfp_1x.py'
    ckpt_file = './work_dirs/cs60_rfp_1x/epoch_7.pth'
    cam = Grad_cam(img_file,configs_files,ckpt_file,if_outfea=True)
    gray_cam = cam()
    show_cam = show_cam_on_img(gray_cam,cam.data_warp)
    cv2.imwrite(os.path.join('./demo/', 'fea_cam' + '.png'), show_cam)