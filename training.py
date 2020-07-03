import torch
from Dataset.MattingDataset import MattingDataset


_TRAIN_FOREGROUND_DIR_ = "./Dataset/Training_set/CombinedForeground"
_TRAIN_BACKGROUND_DIR_ = "./Dataset/Background/COCO_Images"
_TRAIN_ALPHA_DIR_ = "./Dataset/Training_set/CombinedAlpha"

trainingDataset = MattingDataset(_TRAIN_FOREGROUND_DIR_, _TRAIN_BACKGROUND_DIR_, _TRAIN_ALPHA_DIR_)

if __name__ == "__main__":

    compositeImage, trimap = trainingDataset[10]
    print(compositeImage)
    print(f"Dataset length = {len(trainingDataset)}")