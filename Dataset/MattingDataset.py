import os
import torch
import torch.utils.data as data
import itertools

class MattingDataset(data.Dataset):
    def __init__(self, fgDir, bgDir, alphaDir):
        self.fgDir = fgDir
        self.bgDir = bgDir
        self.alphaDir = alphaDir

        self.foregroundImageNames = os.listdir(self.fgDir)
        self.backgroundImageNames = os.listdir(self.bgDir)
        self.alphaImageNames = os.listdir(self.alphaDir)

        self.numForeground = len(self.foregroundImageNames)
        self.numBackground = len(self.backgroundImageNames)
        self.numAlpha = len(self.alphaImageNames)

        assert self.numAlpha == self.numForeground
        
        self.imageBackgroundPair = itertools.product(self.alphaImageNames, self.backgroundImageNames)
        self.imageBackgroundPair = sorted(self.imageBackgroundPair, key=lambda x: x[0])

        assert len(self.imageBackgroundPair) == len(self)

    def __len__(self):
        return self.numAlpha * self.numBackground

    def __getitem__(self, idx):
        alphaMaskName, backgroundName = self.imageBackgroundPair[idx]
        print(alphaMaskName)
        return ([],[])
    