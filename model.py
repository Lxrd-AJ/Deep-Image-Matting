import torch
import torch.nn as nn


class EncoderDecoderNet(nn.Module):
    def __init__(self):
        super(EncoderDecoderNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.finalConv = nn.Conv2d(64, 1, 1, padding=1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        """
        Input to the network is an image patch and the corresponding trimap arranged in the channels of `x`
        """
        x = self.encoder(x)
        print(f"Encoder output size = {x.size()}")
        x = self.decoder(x)
        print(f"Decoder output size = {x.size()}")
        x = self.finalConv(x)
        x = self.sigmoid(x)
        print(x)
        print(f"Network output size = {x.size()}")

        return x


class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()

    def forward(self, x):
        return x



# TODO: Might need to rewrite their AutoEncoder using MathWorks' example https://www.mathworks.com/help/deeplearning/ug/train-a-variational-autoencoder-vae-to-generate-images.html 
# - Is it possible to design an autoencoder that works on any sized input. Are there techniques like global max pooling that can be used?
# use only conv+relu & transconv + relu (decoder), forget maxpool & maxunpool
# Does it matter if it is pretrained??
# encoder output should be 11x11
# http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        """
        The encoder architecture is based off the first 16 layers of VGG16
        """
        self.encoderBlocks = nn.Sequential(
            self.convBatchNormReLU(4, 64, 5),
            self.convBatchNormReLU(64, 128, 3),
            self.convBatchNormReLU(128, 256, 3),
            self.convBatchNormReLU(256, 512, 3, pad=1)
        )


    def convBatchNormReLU(self, inputChannels, outputChannels, kernelSize, pad=1):
        return nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, kernelSize, padding=pad, stride=2, bias=False),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU()
        ) 

    def forward(self, x):
        return self.encoderBlocks(x)




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoderBlocks = nn.Sequential(
            self.transConvBatchNormReLU(512, 256, 3, 2),
            self.transConvBatchNormReLU(256, 128, 3, 2),
            self.transConvBatchNormReLU(128, 64, 1, 2),
            self.transConvBatchNormReLU(64, 64, 1, 2),
        ) 

    def transConvBatchNormReLU(self, inputChannels, outputChannels, kernelSize, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(inputChannels, outputChannels, kernelSize, stride=stride, bias=False),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoderBlocks(x)