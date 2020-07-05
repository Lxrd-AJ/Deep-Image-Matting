import torch
from Dataset.MattingDataset import MattingDataset


_TRAIN_FOREGROUND_DIR_ = "./Dataset/Training_set/CombinedForeground"
_TRAIN_BACKGROUND_DIR_ = "./Dataset/Background/COCO_Images"
_TRAIN_ALPHA_DIR_ = "./Dataset/Training_set/CombinedAlpha"

trainingDataset = MattingDataset(_TRAIN_FOREGROUND_DIR_, _TRAIN_BACKGROUND_DIR_, _TRAIN_ALPHA_DIR_)

def batch_collate_fn(batch):
    """
    - [ ] Add the trimap as the 4th channel  in the input image as the new output image
    - [ ] Randomly crop 320x320 from the output image. Try torchvision.transforms
    - [ ] Or crop the input image to a different size e.g 480x480 and resize it into a 320x320 image
    """
    pass

if __name__ == "__main__":

    compositeImage, trimap = trainingDataset[3201]

    compositeImage.show()
    trimap.show()
    
    print(f"Dataset length = {len(trainingDataset)}")