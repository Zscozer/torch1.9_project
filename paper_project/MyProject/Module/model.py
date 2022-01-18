import torch.nn as nn
from einops import rearrange

import Module.vit as vit
import Module.ResidualBlock as ResidualBlock
from torch.nn import ConvTranspose2d,Conv2d,ReLU,Linear
from einops.layers.torch import Rearrange
from transformers import ViTFeatureExtractor, ViTModel
from einops.layers.torch import Rearrange


class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        # Rerrange
        self.change_size = nn.Sequential(
            Rearrange('b (c h w) -> b c h w', c=3, h=224 // 14, w=224 // 14)
        )
        # Vit-只有一层且没有做预训练
        # self.vit_layer= vit.ViT(image_size=224, patch_size=14,
        #       num_classes=16, dim=768, depth=1, heads=12, mlp_dim=512)


        # self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k') #这个特征提取器要用pil
        self.vit_layer = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # Residual_Block
        self.Res_Block1 = ResidualBlock.ResidualBlock(3,32)
        self.Res_Block2 = ResidualBlock.ResidualBlock(32,128)
        self.Res_Block3 = ResidualBlock.ResidualBlock(128,256)
        self.Res_Block4 = ResidualBlock.ResidualBlock(256,512)
        self.Res_Block5 = ResidualBlock.ResidualBlock(512,256)
        self.Res_Block6 = ResidualBlock.ResidualBlock(256,64)
        # conv
        self.conv = Conv2d(512,512,4,4)
        self.conv2 = Conv2d(512,512,4,2)
        # Deconv
        self.decov1 = ConvTranspose2d(512,512,kernel_size=4,stride=4,padding=0)
        self.decov2 = ConvTranspose2d(512,512,2,2)
        self.decov3 = ConvTranspose2d(512,512,7,7)

        #relu
        self.relu = ReLU()

        # Rearrange
        self.rearrange = nn.Sequential(
            Rearrange('x c h w -> x (c h) w'),
            Rearrange('x n (a b) -> x n a b',a = 5,b=3),
            Rearrange('x n a b -> x (n a) b')
        )
    def forward(self,img):
        # output = self.feature_extractor(img)
        output = self.vit_layer(img)
        output = output.pooler_output
        print("vit_layer",output.shape)
        # output = self.change_size(output)
        output = rearrange(output,'b (c h w) -> b c h w', c=3, h=224 // 14, w=224 // 14)
        print("vit_layer2", output.shape)

        output = self.Res_Block1(output)
        output = self.Res_Block2(output)
        output = self.relu(output)
        output = self.Res_Block3(output)
        output = self.Res_Block4(output)
        output = self.relu(output)
        output = self.conv(output)
        output = self.decov1(output)
        output = self.decov2(output)
        output = self.relu(output)
        # output = self.decov3(output)
        output = self.conv2(output)
        output = self.Res_Block5(output)
        output = self.relu(output)
        output = self.Res_Block6(output)
        # output = self.rearrange(output)
        output = rearrange(output,'x c h w -> x (c h) w')
        output = rearrange(output,'x n (a b) -> x n a b',a = 5,b=3)
        output = rearrange(output,'x n a b -> x (n a) b')
        return output