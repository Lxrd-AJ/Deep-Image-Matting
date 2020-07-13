import torch
import torchvision.transforms as transforms
import random
from PIL import Image
from Dataset.MattingDataset import MattingDataset
from model import EncoderDecoderNet, RefinementNet
from dataset_transforms import RandomTrimapCrop


_TRAIN_FOREGROUND_DIR_ = "./Dataset/Training_set/CombinedForeground"
_TRAIN_BACKGROUND_DIR_ = "./Dataset/Background/COCO_Images"
_TRAIN_ALPHA_DIR_ = "./Dataset/Training_set/CombinedAlpha"

tripleTransforms = transforms.Compose([
    RandomTrimapCrop([(320, 320), (480, 480), (640, 640)], probability=0.9)
])

trainingDataset = MattingDataset(_TRAIN_FOREGROUND_DIR_, _TRAIN_BACKGROUND_DIR_, _TRAIN_ALPHA_DIR_, allTransform=tripleTransforms)

model = EncoderDecoderNet()
refinementModel = RefinementNet(inputChannels=5)

def batch_collate_fn(batch):
    """
    - [ ] Add the trimap as the 4th channel  in the input image as the new output image
    - [ ] Randomly crop 320x320 from the output image. Try torchvision.transforms
    - [ ] Or crop the input image to a different size e.g 480x480 and resize it into a 320x320 image
    """
    pass

def transform_input(data):
    compositeImage, trimap = data
    compositeImage = compositeImage.resize((320,320), Image.BICUBIC)
    trimap = trimap.resize((320,320), Image.BICUBIC)

    # compositeImage.show()
    # trimap.show()
    
    tensorTf = transforms.ToTensor()

    composite = tensorTf(compositeImage)
    trimap = tensorTf(trimap)

    inputTensor = torch.cat([composite,trimap], 0)
    return inputTensor.unsqueeze(0)


if __name__ == "__main__":

    idx = random.randint(0, len(trainingDataset))
    print(f"using image index = {idx}")
    compositeImage, trimap, alphaMask = trainingDataset[idx]

    # alphaMask.show()
    
    print(f"Dataset length = {len(trainingDataset)}")

    x = transform_input((compositeImage, trimap))
    print(f"Composite Input size = {x.size()}")
    
    predictedMask = model(x)

    print(predictedMask)
    print(f"Prediction size = {predictedMask.size()}")

    finalMask = refinementModel(x, predictedMask)
    print(finalMask)
    print(f"Final mask size = {finalMask.size()}")

    predictedMask = transforms.ToPILImage()(finalMask[0] * 255)
    # predictedMask.show()


