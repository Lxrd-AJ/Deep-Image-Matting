import os
import torch
import torch.utils.data as data
import itertools
import numpy as np
import math
import random
from PIL import Image, ImageFilter, ImageChops

class MattingDataset(data.Dataset):
    def __init__(self, fgDir, bgDir, alphaDir, allTransform, imageTransforms):
        self.fgDir = fgDir
        self.bgDir = bgDir
        self.alphaDir = alphaDir

        self.foregroundImageNames = os.listdir(self.fgDir)
        self.backgroundImageNames = os.listdir(self.bgDir)
        random.shuffle(self.backgroundImageNames) #TODO: Remove
        self.backgroundImageNames = self.backgroundImageNames[:3] #TODO: Remove 13
        self.alphaImageNames = os.listdir(self.alphaDir)
        random.shuffle(self.alphaImageNames) #TODO: Remove
        self.alphaImageNames = self.alphaImageNames[:3] #TODO:Remove 23

        self.numForeground = len(self.foregroundImageNames)
        self.numBackground = len(self.backgroundImageNames)
        self.numAlpha = len(self.alphaImageNames)

        # assert self.numAlpha == self.numForeground #TODO: Remove
        
        self.imageBackgroundPair = itertools.product(self.alphaImageNames, self.backgroundImageNames)
        self.imageBackgroundPair = sorted(self.imageBackgroundPair, key=lambda x: x[0])

        self.allTransform = allTransform
        self.imageTransform = imageTransforms

        # assert len(self.imageBackgroundPair) == len(self) #TODO: Remove

    def __len__(self):
        return self.numAlpha * self.numBackground

    def __getitem__(self, idx):
        alphaMaskName, backgroundName = self.imageBackgroundPair[idx]
        alphaMask = self.open_image(os.path.join(self.alphaDir, alphaMaskName)).convert("L")
        foregroundImage = self.open_image(os.path.join(self.fgDir, alphaMaskName))
        backgroundImage = self.open_image(os.path.join(self.bgDir, backgroundName))

        backgroundImage = self.resize_background(backgroundImage, foregroundImage.size)
        
        trimap = self.create_trimap(alphaMask)

        compositeImage = self.composite_image(foregroundImage, backgroundImage, alphaMask)

        assert compositeImage.size == trimap.size, f"composite size = {compositeImage.size} and trimap = {trimap.size} and foreground size = {foregroundImage.size}"

        if self.imageTransform:
            compositeImage = self.imageTransform(compositeImage)
        
        if self.allTransform:
            compositeImage, trimap, alphaMask = self.allTransform((compositeImage, trimap, alphaMask))
        return (compositeImage,trimap, alphaMask)
    
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
        trimap = trimap.convert("L")

        return trimap

    def composite_image(self, foreground, background, alpha):
        bbox = foreground.getbbox()
        fw, fh = foreground.size
        background = background.crop((0,0,fw,fh))
        
        foreground, background, alpha = np.array(foreground), np.array(background), np.array(alpha)
        alphaArr = np.zeros((fh,fw,1), np.float32)
        alphaArr[:,:,0] = alpha / 255
        alpha = alphaArr
        background = alpha * foreground + (1 - alpha) * background
        background = Image.fromarray(background.astype(np.uint8))

        return background
