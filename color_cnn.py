import numpy as np
import torch
import torch.nn as nn
from dncnn import DnCNN
from unet import UNet


class ColorCNN(nn.Module):
    def __init__(self, arch, num_colors, soften=10000, color_norm=1, color_jitter=0):
        super().__init__()
        self.num_colors = num_colors
        self.soften = soften
        self.color_norm = color_norm
        self.color_jitter = color_jitter
        self.base = UNet(3) if arch == 'unet' else DnCNN(3)
        self.color_mask = nn.Sequential(nn.Conv2d(self.base.out_channel, self.base.out_channel * 2, 1), nn.ReLU(),
                                        nn.Conv2d(self.base.out_channel * 2, num_colors, 1, bias=False))
        self.mask_softmax = nn.Softmax2d()

    def forward(self, img, training=False):
        feat = self.base(img)
        m = self.color_mask(feat)
        m = self.mask_softmax(self.soften * m)  # softmax output
        # m.shape = torch.Size([1, 7, 256, 256])

        M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map
        # M.shape = torch.Size([1, 1, 256, 256])

        indicator_M = torch.zeros_like(m).scatter(1, M, 1)
        # indicator_M.shape = torch.Size([1, 7, 256, 256])

        if training:
            color_palette = (img.unsqueeze(1) * m.unsqueeze(2)).sum(dim=[3, 4], keepdim=True) / (
                    m.unsqueeze(2).sum(dim=[3, 4], keepdim=True) + 1e-8) / self.color_norm
            jitter_color_palette = color_palette + self.color_jitter * np.random.randn()
            transformed_img = (m.unsqueeze(2) * jitter_color_palette).sum(dim=2)
        else:
            color_palette = (img.unsqueeze(1) * indicator_M.unsqueeze(2)).sum(dim=[3, 4], keepdim=True) / (
                    indicator_M.unsqueeze(2).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (indicator_M.unsqueeze(2) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette

        # transformed_img ；color quantization results image    m is the weight graph of the original image
        # color_palette


def run():
    img = torch.ones([1, 3, 256, 256])*3
    model = ColorCNN('unet', 7)
    transformed_img = model(img)
    print('transformed_img:', transformed_img[0].size())
    print('m:', transformed_img[1].size())
    print('color_palette:', transformed_img[2].size())
    # transformed_img: torch.Size([1, 7, 256, 256])
    # m: torch.Size([1, 7, 256, 256])
    # color_palette: torch.Size([1, 7, 3, 1, 1])
    pass

# run()
