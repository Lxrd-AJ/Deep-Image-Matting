import torch
import torch.nn as nn


class RefinementNet(nn.Module):
    def __init__(self, inputChannels):
        super(RefinementNet, self).__init__()
        self.layers = nn.Sequential(
            convBatchNormReLU(inputChannels, 64, 3, pad=1, stride=1),
            convBatchNormReLU(64, 64, 3, pad=1, stride=1),
            convBatchNormReLU(64, 64, 3, pad=1, stride=1),
            nn.Conv2d(64, 1, 3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    """
    - The input x currently contains the image+trimap. It is then concatenated with the predictedmask. 
        - [ ] Do I need only the image+predictedMask ? Currently i have image+trimap+predictedMask as input to the refinement net
    """
    def forward(self, x, predictedMask):
        x = torch.cat([x,predictedMask], 1)
        x = self.layers(x)
        x = self.sigmoid(x)
        return x



class EncoderDecoderNet(nn.Module):
    def __init__(self):
        super(EncoderDecoderNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        """
        Input to the network is an image patch and the corresponding trimap arranged in the channels of `x`
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        """
        The encoder architecture is based off the first 16 layers of VGG16
        """
        self.encoderBlocks = nn.Sequential(
            convBatchNormReLU(4, 64, 3),
            convBatchNormReLU(64, 64, 1, pad=0, stride=1),
            convBatchNormReLU(64, 128, 3),
            convBatchNormReLU(128, 128, 1, pad=0, stride=1),
            convBatchNormReLU(128, 256, 3),
            convBatchNormReLU(256, 256, 1, pad=0, stride=1),
            convBatchNormReLU(256, 512, 3),
            convBatchNormReLU(512, 512, 1, pad=0, stride=1),
        )

    def forward(self, x):
        return self.encoderBlocks(x)



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoderBlocks = nn.Sequential(
            self.transConvBatchNormReLU(512, 256, 3, 2),
            self.transConvBatchNormReLU(256, 128, 3, 2),
            self.transConvBatchNormReLU(128, 64, 3, 2),
            self.transConvBatchNormReLU(64, 1, 3, 2, outPad=1),
        ) 

    def transConvBatchNormReLU(self, inputChannels, outputChannels, kernelSize, stride, outPad=0):
        return nn.Sequential(
            nn.ConvTranspose2d(inputChannels, outputChannels, kernelSize, stride=stride, bias=False, output_padding=outPad),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoderBlocks(x)


def convBatchNormReLU(inputChannels, outputChannels, kernelSize, pad=0, stride=2):
        return nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, kernelSize, padding=pad, stride=stride, bias=False),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU()
        )