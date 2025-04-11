# from __future__ import print_function
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from color_cnn import ColorCNN
from dataloader import MyDataset
import cv2
import pandas as pd
from tools import reconst_loss, squared_mahalanobis_distance_loss, mono_color_reconst_loss

run_name = '2016to2012'
batch_size = 12
seed = 1
num_workers = 4
reconst_loss_type = 'l1'
save_layer_train = 12
log_interval = 100

try:
    os.makedirs(f'./train_results/{run_name}')
except OSError:
    pass

torch.manual_seed(seed)
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

csv_path = 'sample.csv'
num_primary_color = 7
train_dataset = MyDataset(csv_path, num_primary_color, mode='train')

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False,
    pin_memory=True
)

val_dataset = MyDataset(csv_path, num_primary_color, mode='val')
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

def read_backimage():
    img = cv2.imread('./backimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32))  # torch.Size([3, 256, 256])
    return img.view(1, 3, 256, 256).to(device)


backimage = read_backimage()
# torch.Size([1, 3, 256, 256])

def alpha_normalize(alpha_layers):
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

# color_CNN
plan2_generator = ColorCNN('unet', num_primary_color).to(device)

# alpha predictor
mask_generator = MaskGenerator(num_primary_color).to(device)

# residual predictor
residue_predictor = ResiduePredictor(num_primary_color).to(device)


params = list(plan2_generator.parameters())
params += list(mask_generator.parameters())
params += list(residue_predictor.parameters())

optimizer = optim.Adam(params, lr=0.0002, betas=(0.0, 0.99))

def train(epoch, batch_loss, best_now):
    plan2_generator.train()
    mask_generator.train()
    residue_predictor.train()

    best_performance = best_now

    for batch_idx, (target_img) in enumerate(train_loader):
        target_img = target_img.to(device)   # torch.Size([12, 3, 256, 256])  (batchsize, c, h, w)
        optimizer.zero_grad()
        transformed_img, prob, color_results = plan2_generator(target_img, training=True)
        # prob.shape = torch.Size([12, 7, 256, 256])
        # color_results.shape = torch.Size([12, 7, 3, 1, 1])

        prob_max = torch.var(prob.view([target_img.size(0), 7, -1]), dim=2)
        # torch.var calculates the variance of the tensor
        avg_max = torch.mean(prob_max)
        regular_loss = np.log2(7) * (1 - avg_max)
        # Used to calculate a regularization loss

        plate_color_layers = color_results.repeat(1, 1, 1, target_img.size(2), target_img.size(3)) # repeat
        # shape = torch.Size([12, 7, 3, 256, 256])

        primary_color_pack = plate_color_layers.view(target_img.size(0), -1, target_img.size(2), target_img.size(3))
        # shape = torch.Size([12, 21, 256, 256])

        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        # shape = torch.Size([12, 7, 256, 256])

        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
        # shape = torch.Size([12, 7, 1, 256, 256])

        # Alpha channel standardization
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)

        mono_color_layers = torch.cat((plate_color_layers, processed_alpha_layers), 2)
        #shape: (12, 7, 4, 256, 256)

        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1, target_img.size(2), target_img.size(3))
        # shape = torch.Size([12, 28, 256, 256])

        residue_pack = residue_predictor(target_img, mono_color_layers_pack)

        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        pred_unmixed_rgb_layers = torch.clamp((plate_color_layers + residue), min=0., max=1.0)
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)

        mono_color_reconst_img = (plate_color_layers * processed_alpha_layers).sum(dim=1)

        r_loss = reconst_loss(reconst_img, target_img, type=reconst_loss_type)  # Lr
        m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img)  # Lm is responsible for the alpha layer
        d_loss = squared_mahalanobis_distance_loss(plate_color_layers.detach(), processed_alpha_layers,
                                                   pred_unmixed_rgb_layers)
        # Ld:In order to collect only homogeneous colors in each color layer

        total_loss = 7 * r_loss + 10 * m_loss + 2 * d_loss + 1 * regular_loss
        total_loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0 and batch_idx != 0:
            if (total_loss.item() / len(target_img)) < best_performance:
                best_performance = (total_loss.item() / len(target_img))
                torch.save(plan2_generator.state_dict(), 'train_results/%s/new_model/plan2_generator.pth' % run_name)
                torch.save(mask_generator.state_dict(), 'train_results/%s/new_model/mask_generator.pth' % run_name)
                torch.save(residue_predictor.state_dict(), 'train_results/%s/new_model/residue_predictor.pth' % run_name)

            for save_layer_number in range(save_layer_train):
                save_image(plate_color_layers[save_layer_number],
                       'train_results/%s/train_ep_' % run_name + str(epoch) + '_ln_%02d_plate_color_layers.png' % save_layer_number)
                save_image(pred_unmixed_rgb_layers[save_layer_number] * processed_alpha_layers[save_layer_number] + (1 - processed_alpha_layers[save_layer_number]),
                       'train_results/%s/train_ep_' % run_name + str(epoch) + '_ln_%02d_pred_unmixed_rgb_layers.png' % save_layer_number)
                save_image(reconst_img[save_layer_number].unsqueeze(0),
                       'train_results/%s/train_ep_' % run_name + str(epoch) + '_ln_%02d_reconst_img.png' % save_layer_number)
                save_image(mono_color_reconst_img[save_layer_number].unsqueeze(0),
                       'train_results/%s/train_ep_' % run_name + str(epoch) + '_ln_%02d_mono_color_reconst_img.png' % save_layer_number)
                save_image(target_img[save_layer_number].unsqueeze(0),
                       'train_results/%s/train_ep_' % run_name + str(epoch) + '_ln_%02d_target_img.png' % save_layer_number)
            batch_loss.append(total_loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)] ---> Total_loss:{:.6f}'.format(
                epoch, batch_idx * len(target_img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss.item()
            ))

    # save model
    torch.save(plan2_generator.state_dict(), 'train_results/%s/new_model/plan2_generator_last.pth' % run_name)
    torch.save(mask_generator.state_dict(), 'train_results/%s/new_model/mask_generator_last.pth' % run_name, _use_new_zipfile_serialization=False)
    torch.save(residue_predictor.state_dict(), 'train_results/%s/new_model/residue_predictor_last.pth' % run_name, _use_new_zipfile_serialization=False)

    return batch_loss, best_performance


if __name__ == '__main__':
    epochs = 100
    batch_loss = []
    best_now = 10
    for epoch in range(1, epochs + 1):
        print('Start training')
        batch_loss, best_now = train(epoch, batch_loss, best_now)
    log = {"total_loss": batch_loss}
