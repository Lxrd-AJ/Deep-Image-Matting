import os
import torch
import torch.utils.data as data
import itertools
import numpy as np
import math
import random
from PIL import Image, ImageFilter, ImageChops

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
        alphaMask = self.open_image(os.path.join(self.alphaDir, alphaMaskName))
        foregroundImage = self.open_image(os.path.join(self.fgDir, alphaMaskName))
        backgroundImage = self.open_image(os.path.join(self.bgDir, backgroundName))

        backgroundImage = self.resize_background(backgroundImage, foregroundImage.size)
        
        trimap = self.create_trimap(alphaMask)

        compositeImage = self.composite_image(foregroundImage, backgroundImage, alphaMask)
        
        return (compositeImage,trimap)
    
    def open_image(self, path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def resize_background(self, backgroundImage, foregroundSize):
        """
        The background image could have a different size and aspect 
        ratio to the foreground image.
        Therefore, the background needs to be transformed into the 
        same size as the foreground
        """
        fw, fh = foregroundSize
        bw, bh = backgroundImage.size
        wratio = fw / bw
        hratio = fh / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            newWidth, newHeight = math.ceil(bw*ratio), math.ceil(bh*ratio)
            backgroundImage = backgroundImage.resize((newWidth,newHeight), Image.BICUBIC)
        return backgroundImage

    def create_trimap(self, alphaMask):
        segmentedAlpha = np.array(alphaMask)
        segmentedAlpha[segmentedAlpha > 0] = 255
        segmentedAlpha = Image.fromarray(segmentedAlpha)

        # NB: Dilation & Erosion seem to be time consuming operations
        dilationValues = [7,9,11] #,13,15
        erosionValues = [3,5,7] #,9,11
        dv = random.choice(dilationValues)
        ev = random.choice(erosionValues)
        erodedAlpha = segmentedAlpha.filter(ImageFilter.MinFilter(ev))
        dilatedMask = segmentedAlpha.filter(ImageFilter.MaxFilter(dv))
        # middle step: threshold the dilatedMask to 127
        dilatedMask = np.array(dilatedMask)
        dilatedMask[dilatedMask > 127] = 127
        dilatedMask = Image.fromarray(dilatedMask)
        trimap = ImageChops.add(dilatedMask, erodedAlpha)

        return trimap

    def composite_image(self, foreground, background, alpha):
        bbox = foreground.getbbox()
        background = background.crop(bbox)
        alpha = alpha.convert("1")
        background = ImageChops.composite(foreground, background,alpha)
        return background
