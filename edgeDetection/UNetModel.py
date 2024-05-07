'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk/ afjsliny@gmail.com
Date: 2024-03-18 20:03:20
LastEditTime: 2024-03-18 20:09:31
Description: 
'''
import torch
import torch.nn as nn

class UNet(nn.Module):
    pass
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # more conv layers and pooling layers
        )

        self.decoder = nn.Sequential(

            # more conv layers and upsampling layers
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=1),
            nn.Sigmoid()    # output the probability of the edge
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
model = UNet()