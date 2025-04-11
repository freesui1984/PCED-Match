from tools import cut_edge, alpha_normalize, proc_guidedfilter
import torch.utils.data
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from color_cnn import ColorCNN
import os
import numpy as np
from skimage.exposure import match_histograms
from skimage.io import imread

csv_path = 'sample.csv'
num_primary_color = 7

path_plan2_generator = 'train_results/2016to2012/new_model/plan2_generator.pth'
path_mask_generator = 'train_results/2016to2012/new_model/mask_generator.pth'
path_residue_predictor = 'train_results/2016to2012/new_model/residue_predictor.pth'

device = 'cuda'
# define model
plan2_generator = ColorCNN('unet', num_primary_color).to(device)
mask_generator = MaskGenerator(num_primary_color).to(device)
residue_predictor = ResiduePredictor(num_primary_color).to(device)
# load params
plan2_generator.load_state_dict(torch.load(path_plan2_generator))
mask_generator.load_state_dict(torch.load(path_mask_generator))
residue_predictor.load_state_dict(torch.load(path_residue_predictor))
# eval mode
plan2_generator.eval()
mask_generator.eval()
residue_predictor.eval()

def read_dataimage(path):
    img = imread(path)
    img = img.transpose((2, 0, 1))
    target_img = img / 255  # 0~1

    # to Tensor
    target_img = torch.from_numpy(target_img.astype(np.float32))
    target_img = target_img.unsqueeze(0)
    return target_img  # return torch.Tensor

print('Start!')

batch_idx = 0
img_list = [0] * 2

with torch.no_grad():
    reference_folder = "dataset/building_change_detection/2012/test/images"
    source_folder = "dataset/building_change_detection/2016/test/images"
    output_folder = "test_results/images"
    for filename in os.listdir(source_folder):
        batch_idx += 1
        source_path = os.path.join(source_folder, filename)
        ref_path = os.path.join(reference_folder, filename)
        matched_path = os.path.join(output_folder, filename)
        print('img #', batch_idx)
        img_list[0] = ref_path
        img_list[1] = source_path
        for i in range(len(img_list)):
            target_img = read_dataimage(img_list[i])
            target_img = target_img.to(device)
            # 颜色提取模块
            a, b, color_results = plan2_generator(target_img, training=False)
            # torch.Size([1, 7, 3, 1, 1])
            color_results1 = color_results.repeat(color_results.size(0), 1, 1, target_img.size(2), target_img.size(3))
            # torch.Size([1, 7, 3, 512, 512])

            plate_color_pack = color_results1.view(color_results1.size(0), -1, color_results1.size(3),
                                                   color_results1.size(4))
            plate_color_pack = cut_edge(plate_color_pack)
            # torch.Size([1, 21, 512, 512])

            plate_color_layers = plate_color_pack.view(plate_color_pack.size(0), -1, 3, plate_color_pack.size(2),
                                                       plate_color_pack.size(3))
            # torch.Size([1, 7, 3, 512, 512])

            # 颜色分解模块——alpha预测器
            pred_alpha_layers_pack = mask_generator(target_img, plate_color_pack)
            # torch.Size([1, 7, 512, 512])

            pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2),
                                                            target_img.size(3))
            # torch.Size([1, 7, 1, 512, 512])

            ## Alpha Layer Proccessing
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img)
            processed_alpha_layers = alpha_normalize(processed_alpha_layers)
            # torch.Size([1, 7, 1, 512, 512])

            if i == 0:
                alpha_ref = processed_alpha_layers.clone().cpu().numpy().transpose(0, 1, 3, 4, 2)
                # (1, 7, 512, 512, 1)
                alpha_ref = alpha_ref[0, :, :, :, :]
            elif i == 1:
                alpha_source = processed_alpha_layers.clone().cpu().numpy().transpose(0, 1, 3, 4, 2)
                # size(1, 7, 512, 512, 1)
                alpha_source = alpha_source[0, :, :, :, :]  # size(7, 512, 512, 1)

            mono_color_layers = torch.cat((plate_color_layers, processed_alpha_layers), 2)
            # torch.Size([1, 7, 4, 512, 512])

            mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1, target_img.size(2), target_img.size(3))
            # torch.Size([1, 28, 512, 512])

            # 颜色分解模块——残差预测器
            residue_pack = residue_predictor(target_img, mono_color_layers_pack)
            # torch.Size([1, 21, 512, 512])

            residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
            # torch.Size([1, 7, 3, 512, 512])

            pred_unmixed_rgb_layers = torch.clamp((plate_color_layers + residue), min=0., max=1.0)
            # torch.Size([1, 7, 3, 512, 512])

            if i == 0:
                rgb_save = pred_unmixed_rgb_layers.clone().cpu().numpy().transpose(0, 1, 3, 4, 2)
                rgb_ref = np.squeeze(rgb_save, axis=0)  # size(7, 512, 512, 3)
            elif i == 1:
                rgb_save = pred_unmixed_rgb_layers.clone().cpu().numpy().transpose(0, 1, 3, 4, 2)
                rgb_source = np.squeeze(rgb_save, axis=0)  # size(7, 512, 512, 3)
                matched_layer = np.ones_like(rgb_source)

                # Each layer of the source image is histogram matched with the corresponding layer in the reference image
                for j in range(7):
                    matched_layer[j, :, :, :] = match_histograms(rgb_source[j, :, :, :], rgb_ref[j, :, :, :], channel_axis=2)

                matched_layer = torch.from_numpy(matched_layer.transpose(0, 3, 1, 2))

                processed_alpha_layers = processed_alpha_layers.clone().cpu()
                matched_image = matched_layer * processed_alpha_layers[0, :, :, :, :]
                matched_image = matched_image.sum(dim=0)
                save_image(matched_image[:, :, :], matched_path)
