import numpy as np
import random
from PIL import Image

class RandomTrimapCrop(object):
    def __init__(self, sizeRange, probability=0.5):
        self.sizeRange = sizeRange
        self.p = probability

    """
    Given a list of images in items
    """
    def __call__(self, items):
        image, trimap, mask = items
        image.show()
        trimap.show()
        mask.show()

        if random.random() < self.p:
            cropWidth, cropHeight = random.choice(self.sizeRange)
            print(f"Chosen Crop Size = {(cropWidth, cropHeight)}")

            trimapArray = np.array(trimap)
            unknownIndices = np.where(trimapArray == 127)
            unknownIndices = list(zip(unknownIndices[0], unknownIndices[1])) #row,col => height, width
            
            if len(unknownIndices) > 0:
                y,x = random.choice(unknownIndices)
                print(f"Chosen center = {x},{y}")
                print(trimapArray[y,x])

                topLeftx = max(0, x - int(cropWidth/2))
                topLefty = max(0, y - int(cropHeight/2))

                print(f"Top Left x,y = {topLeftx}, {topLefty}")

                #Crop image
                image = self.crop(image, (topLeftx, topLefty), (cropWidth,cropHeight))
                image.show()

                #Trimap Crop
                trimap = self.crop(trimap, (topLeftx, topLefty), (cropWidth,cropHeight))
                trimap.show()
                mask = self.crop(mask, (topLeftx, topLefty), (cropWidth,cropHeight))
                mask.show()
            #

        return image, trimap, mask

    def crop(self, img, topLeft, size):
        x,y = topLeft
        w,h = size
        imgArr = np.array(img)
        imgArrCrop = imgArr[y:y+h, x:x+w]
        img = Image.fromarray(imgArrCrop)
        return img