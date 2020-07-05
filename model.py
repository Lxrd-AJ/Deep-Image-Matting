import torch
import torch.nn as nn

class EncoderDecoderNet(nn.Module):
    def __init__(self):
        """
        The encoder architecture is based off the first 16 layers of VGG16
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),


            # 128
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            ## 256
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),


            # 512
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),


            # 512
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),


            nn.Conv2d(512, 1024, 3, padding=1, bias=False)
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )


        """
        - 6 conv (transpose conv) layers
        - 5 unpooling layers (MaxUnpool2d)
        - final alpha prediction layer (conv)
        """
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(512, 512, 5, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(512, 256, 5, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(256, 128, 5, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(128, 64, 5, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(64, 64, 5, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 5, padding=1, bias=False),
            
        )

    def forward(self, x):
        """
        Input to the network is an image patch and the corresponding trimap arranged in the channels of `x`
        """
        pass

class RefinementNet(nn.Module):
    def __init__(self):
        pass