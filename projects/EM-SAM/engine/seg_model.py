import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .build_sam import sam_model_registry
from connectomics.model.utils.misc import IntermediateLayerGetter

backbone_param_dict = {
    'vit_b': [12,768],
    'vit_l': [24,1024],}

def load_Encoder(arch = "vit_b", checkpoint_path = ""):
    bb_arch, bb_size = arch.split('_')[0], arch.split('_')[1]+'_'+arch.split('_')[2]
    print("model arch:", bb_arch, "model size:", bb_size)
    if bb_arch == 'sam':
        checkpoint_path = "./sam_checkpoints/sam_vit_b_01ec64.pth"
        backbone = sam_model_registry[arch.split('_')[1]+'_'+arch.split('_')[2]](checkpoint=checkpoint_path)
        feat_keys = [None] * 2
        return_layers = {'patch_embed': feat_keys[0],
                         'blocks': feat_keys[1],}
                         # 'neck': feat_keys[2]}

        return IntermediateLayerGetter(backbone.image_encoder, return_layers)

class unetr(nn.Module):
    def __init__(self, 
                 img_shape, 
                 input_dim, 
                 output_dim,
                 embed_dim, 
                 patch_size, 
                 dropout,
                 arch: str, 
                 checkpoints: str,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        self.patch_dim = [int(img_shape[0]//patch_size), int(img_shape[1]//patch_size)]
        self.backbone = load_Encoder(arch,checkpoints)


        depth,self.embed_dim = backbone_param_dict[arch.split('_')[1]+'_'+arch.split('_')[2]]
        self.depth = depth//4
        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv2DBlock(input_dim, 32, 3),
                Conv2DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv2DBlock(self.embed_dim, 512),
                Deconv2DBlock(512, 256),
                Deconv2DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv2DBlock(self.embed_dim, 512),
                Deconv2DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv2DBlock(self.embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv2DBlock(self.embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv2DBlock(1024, 512),
                Conv2DBlock(512, 512),
                Conv2DBlock(512, 512),
                SingleDeconv2DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2DBlock(512, 256),
                Conv2DBlock(256, 256),
                SingleDeconv2DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2DBlock(256, 128),
                Conv2DBlock(128, 128),
                SingleDeconv2DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2DBlock(128, 64),
                Conv2DBlock(64, 64),
                SingleConv2DBlock(64, output_dim, 1)
            )
            
    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        z = self.backbone.patch_embed(x)
        feat = []
        for layers in self.backbone.blocks:
            z = layers(z)
            feat.insert(0,z)
        ext_feat = []
        for i in range(1,5):
            ext_feat.insert(0, feat[i * self.depth - 1])
        z0, z3, z6, z9, z12 = x, *ext_feat

        shape_len = len(z3.shape)
        if shape_len == 4:
            z3 = z3.permute(0, 3, 1, 2)
            z6 = z6.permute(0, 3, 1, 2)
            z9 = z9.permute(0, 3, 1, 2)
            z12 = z12.permute(0, 3, 1, 2)
        elif shape_len == 3:
            z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
            z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
            z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
            z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)

class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)

class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)