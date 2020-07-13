import numpy as np
import random

class RandomCrop(object):
    def __init__(self, sizeRange):
        self.sizeRange = sizeRange

    """
    Given a list of images in items
    """
    def __call__(self, items):
        image, trimap, mask = items
        cropWidth, cropHeight = random.choice(self.sizeRange)
        print(f"Chosen Crop Size = {(cropWidth, cropHeight)}")

        trimapArray = np.array(trimap)
        unknownIndices = np.where(trimapArray == 127)
        unknownIndices = list(zip(unknownIndices[0], unknownIndices[1])) #row,col => height, width
        
        if len(unknownIndices) > 0:
            y,x = random.choice(unknownIndices)
            print(f"Chosen center = {x},{y}")
            print(trimapArray[y,x])

            x = max(0, x - int(cropWidth/2))
            y = max(0, y - int(cropHeight/2))

            print(f"Center x,y = {x}, {y}")

        return image, trimap, mask