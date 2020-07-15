import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class ToTensor(object):
    def __call__(self, items):
        image, trimap, mask = items

        image = TF.to_tensor(image)

        trimap = np.array(trimap)
        trimap = torch.from_numpy(trimap).float() / 255.0
        
        mask = np.array(mask)
        mask = torch.from_numpy(mask).float() / 255

        return (image,  trimap, mask)



class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, items):
        return tuple(self.resize(x) for x in items)

    def resize(self, x):
        return x.resize(self.size, Image.BICUBIC)



class RandomTrimapCrop(object):
    """
    Crops the input image, trimap and alpha mask into a size chosen randomly from `sizeRange`
    The center (x,y) of the returned results is chosen from a random location in the unknown regions of the trimap
    """
    def __init__(self, sizeRange, probability=0.5):
        self.sizeRange = sizeRange
        self.p = probability

    """
    
    """
    def __call__(self, items):
        image, trimap, mask = items

        if random.random() < self.p:
            cropWidth, cropHeight = random.choice(self.sizeRange)

            trimapArray = np.array(trimap)
            unknownIndices = np.where(trimapArray == 127)
            unknownIndices = list(zip(unknownIndices[0], unknownIndices[1])) #row,col => height, width
            
            if len(unknownIndices) > 0:
                y,x = random.choice(unknownIndices)

                topLeftx = max(0, x - int(cropWidth/2))
                topLefty = max(0, y - int(cropHeight/2))

                image = self.crop(image, (topLeftx, topLefty), (cropWidth,cropHeight))
                trimap = self.crop(trimap, (topLeftx, topLefty), (cropWidth,cropHeight))
                mask = self.crop(mask, (topLeftx, topLefty), (cropWidth,cropHeight))
                

        return image, trimap, mask

    def crop(self, img, topLeft, size):
        x,y = topLeft
        w,h = size
        imgArr = np.array(img)
        imgArrCrop = imgArr[y:y+h, x:x+w]
        img = Image.fromarray(imgArrCrop)
        return img