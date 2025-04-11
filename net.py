import torch
import torch.nn as nn
import torch.nn.functional as F

# Alpha预测器
class MaskGenerator(nn.Module):

    def __init__(self, num_primary_color):
        super(MaskGenerator, self).__init__()
        in_dim = 3 + num_primary_color * 3   # 24ch (= 3 + 7 * 3)
        out_dim = num_primary_color     # num_out_layers is the same as num_primary_color.

        self.conv1 = nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_dim * 8, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.conv4 = nn.Conv2d(in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, primary_color_pack):
        # target_img.shape = torch.Size([12, 3, 256, 256])
        # primary_color_pack.shape = torch.Size([12, 21, 256, 256])
        # print("net-33:", primary_color_pack.size())
        # print("net-34:", target_img.size())
        x = torch.cat((target_img, primary_color_pack), dim=1)
        # print("net-36:", x.shape)
        # x.shape = torch.Size([12, 24, 256, 256])

        h1 = self.bn1(F.relu(self.conv1(x))) # (N,C*2,H/2,W/2)
        h2 = self.bn2(F.relu(self.conv2(h1))) # (N,C*4,H/4,W/4)
        h3 = self.bn3(F.relu(self.conv3(h2))) # (N,C*8,H/8,W/8)
        h4 = self.bnde1(F.relu(self.deconv1(h3))) # (N,C*4,H/4,W/4)
        h4 = torch.cat((h4, h2), 1) # (N,C*8,H/4,W/4)
        h5 = self.bnde2(F.relu(self.deconv2(h4))) # (N,C*2,H/2,W/2)
        h5 = torch.cat((h5, h1), 1) # (N,C*4,H/2,W/2)
        h6 = self.bnde3(F.relu(self.deconv3(h5))) # (N,C*2,H,W)
        h6 = torch.cat((h6, target_img), 1) # (N,C*2+3,H,W)
        h7 = self.bn4(F.relu(self.conv4(h6)))  # (N,C,H,W)

        return torch.sigmoid(self.conv5(h7)) # box constraint for alpha layers
        # (N,out_dim,H,W)

# 残差预测器
class ResiduePredictor(nn.Module):
    def __init__(self, num_primary_color):
        super(ResiduePredictor, self).__init__()

        in_dim = 3 + num_primary_color * 4  # RGBA,四通道7个颜色层
        out_dim = num_primary_color * 3

        self.conv1 = nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_dim * 8, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.conv4 = nn.Conv2d(in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, mono_color_layers_pack):
        # target_img   Size([12, 3, 256, 256])
        # mono_color_layers_pack   Size([12, 28, 256, 256])
        # caution: mono_color_layers_pack is different from primary_color_pack.
        x = torch.cat((target_img, mono_color_layers_pack), dim=1)
        # torch.Size([12, 31, 256, 256])

        h1 = self.bn1(F.relu(self.conv1(x)))   # (N,C*2,H/2,W/2)
        h2 = self.bn2(F.relu(self.conv2(h1)))   # (N,C*4,H/4,W/4)
        h3 = self.bn3(F.relu(self.conv3(h2)))   # (N,C*8,H/8,W/8)
        h4 = self.bnde1(F.relu(self.deconv1(h3)))   # *4
        h4 = torch.cat((h4, h2), 1)   # *8
        h5 = self.bnde2(F.relu(self.deconv2(h4)))   # *2
        h5 = torch.cat((h5, h1), 1)   # *4
        h6 = self.bnde3(F.relu(self.deconv3(h5)))   # *2
        h6 = torch.cat((h6, target_img), 1)   # *2+3
        h7 = self.bn4(F.relu(self.conv4(h6)))

        residue_pack = torch.tanh(self.conv5(h7)) # box constraint for alpha layers
        # 各レイヤーの各RGBチャンネルごとに，residueの平均が0になるように正規化する
        residue_pack = residue_pack - residue_pack.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        return residue_pack # residue(pack) only
