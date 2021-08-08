import torch
import numpy as np
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from .blocks import DoubleConvolutionBlock, UpSampleBlock


class MyUNet(nn.Module):

    def __init__(
        self, image_height: int, image_width: int,
        n_classes: int, efficientnet_alias: str = 'b0'):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained(
            'efficientnet-{}'.format(efficientnet_alias)
        )
        self.image_height = image_height
        self.image_width = image_width
        self.conv0 = DoubleConvolutionBlock(5, 64)
        self.conv1 = DoubleConvolutionBlock(64, 128)
        self.conv2 = DoubleConvolutionBlock(128, 512)
        self.conv3 = DoubleConvolutionBlock(512, 1024)
        self.mp = nn.MaxPool2d(2)
        self.up1 = UpSampleBlock(1282 + 1024, 512)
        self.up2 = UpSampleBlock(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)
    
    def get_mesh(self, batch_size, shape_x, shape_y):
        mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
        mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
        mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
        mesh = torch.cat([torch.tensor(mg_x), torch.tensor(mg_y)], 1)
        return mesh

    def forward(self, x):
        batch_size = x.shape[0]
        mesh_1 = self.get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh_1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        x_center = x[:, :, :, self.image_width // 8: -self.image_width // 8]
        features = self.base_model.extract_features(x_center)
        back_ground = torch.zeros([
            features.shape[0], features.shape[1], features.shape[2], features.shape[3] // 8])
        features = torch.cat([back_ground, features, back_ground], 3)
        mesh_2 = self.get_mesh(batch_size, features.shape[2], features.shape[3])
        features = torch.cat([features, mesh_2], 1)
        x = self.up1(features, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x
