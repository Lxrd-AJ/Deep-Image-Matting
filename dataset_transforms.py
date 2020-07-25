import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

class RandomAffine(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, items):
        image, trimap, mask = items
        if random.random() < self.p:
            angle = random.randint(-180, 180)
            image = TF.affine(image, angle, translate=[0,0], scale=1.0, shear=0, resample=Image.BICUBIC)
            # use nearest so the values of the trimap and alpha mask are not changed
            trimap = TF.affine(trimap, angle, translate=[0,0], scale=1.0, shear=0, resample=Image.NEAREST)
            mask = TF.affine(mask, angle, translate=[0,0], scale=1.0, shear=0, resample=Image.NEAREST)
        return image, trimap, mask

class RandomBlur(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        return image

class RandomRotation(object):
    def __init__(self, probability=0.5, angle=45):
        self.p = probability
        self.angle = angle

    def __call__(self, items):
        image, trimap, mask = items
        angle = random.randint(-self.angle, self.angle)
        if random.random() < self.p:
            image = TF.rotate(image, angle)
            trimap = TF.rotate(trimap, angle)
            mask = TF.rotate(mask, angle)
        return image, trimap, mask

class RandomVerticalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, items):
        image, trimap, mask = items
        if random.random() < self.p:
            image = TF.vflip(image)
            trimap = TF.vflip(trimap)
            mask = TF.vflip(mask)
        return image, trimap, mask

class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, items):
        image, trimap, mask = items
        if random.random() < self.p:
            image = TF.hflip(image)
            trimap = TF.hflip(trimap)
            mask = TF.hflip(mask)
        return image, trimap, mask

class ToTensor(object):
    def __call__(self, items):
        image, trimap, mask = items

        image = TF.to_tensor(image)

        trimap = np.array(trimap)
        trimap = torch.from_numpy(trimap).float() / 255
        
        mask = np.array(mask)
        mask = torch.from_numpy(mask).float() / 255

        return (image,  trimap, mask)



class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, items):
        return tuple(self.resize(x) for x in items)

    def resize(self, x):
        # Using the NEAREST filter leaves the trimap pixels unchanged
        # A filter like BICUBIC would result in the trimap having values other than 0, 127 & 255
        # as it can interpolate pixel values between 0 and 255
        return x.resize(self.size, Image.NEAREST)



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