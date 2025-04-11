import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.nn import functional as F
import torch
import numpy as np
import cv2
from guided_filter_pytorch.guided_filter import GuidedFilter

# loss
def reconst_loss(reconst_img, target_img, type='mse'):
    if type == 'mse':
        loss = F.mse_loss(reconst_img, target_img.detach())
    elif type == 'l1':
        loss = F.l1_loss(reconst_img, target_img.detach())
    elif type == 'vgg':
        pass
    return loss

def temp_distance(primary_color_layers, alpha_layers, rgb_layers):
    diff = (primary_color_layers - rgb_layers)
    distance = (diff * diff).sum(dim=2, keepdim=True) # out: (bn, ln, 1, h, w)
    loss = (distance * alpha_layers).sum(dim=1, keepdim=True).mean()
    return loss # shape = (bn)

def squared_mahalanobis_distance_loss(primary_color_layers, alpha_layers, rgb_layers):
    loss = temp_distance(primary_color_layers, alpha_layers, rgb_layers)
    return loss

def mono_color_reconst_loss(mono_color_reconst_img, target_img):
    # loss = F.l1_loss(mono_color_reconst_img, target_img.detach())
    loss = F.mse_loss(mono_color_reconst_img, target_img.detach())
    return loss

def replace_color(primary_color_layers, manual_colors):
    temp_primary_color_layers = primary_color_layers.clone()
    for layer in range(len(manual_colors)):
        for color in range(3):
            temp_primary_color_layers[:, layer, color, :, :].fill_(manual_colors[layer][color])
    return temp_primary_color_layers

def cut_edge(target_img):
    # print(target_img.size())
    target_img = F.interpolate(target_img, scale_factor=1, mode='area')
    # print(target_img.size())
    h = target_img.size(2)
    w = target_img.size(3)
    h = h - (h % 8)
    w = w - (w % 8)
    target_img = target_img[:, :, :h, :w]
    # print(target_img.size())
    return target_img

def alpha_normalize(alpha_layers):
    # constraint (sum = 1)
    # layersの状態で受け取り，その形で返す. bn, ln, 1, h, w
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

def read_backimage():
    img = cv2.imdecode(np.fromfile('./backimage.jpg', dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = img / 255
    img = torch.from_numpy(img.astype(np.float32))

    return img.view(1, 3, 256, 256).to('cuda')

backimage = read_backimage()

def proc_guidedfilter(alpha_layers, guide_img):
    guide_img = (guide_img[:, 0, :, :] * 0.299 + guide_img[:, 1, :, :] * 0.587 + guide_img[:, 2, :, :] * 0.114).unsqueeze(1)

    for i in range(alpha_layers.size(1)):
        layer = alpha_layers[:, i, :, :, :]

        processed_layer = GuidedFilter(3, 1 * 1e-6)(guide_img, layer)
        if i == 0:
            processed_alpha_layers = processed_layer.unsqueeze(1)
        else:
            processed_alpha_layers = torch.cat((processed_alpha_layers, processed_layer.unsqueeze(1)), dim=1)

    return processed_alpha_layers