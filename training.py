import torch
import torchvision.transforms as transforms
import random
import numpy as np
import multiprocessing
import time
from PIL import Image
from Dataset.MattingDataset import MattingDataset
from model import EncoderDecoderNet, RefinementNet
from dataset_transforms import RandomTrimapCrop, Resize, ToTensor
from loss import alpha_prediction_loss




def batch_collate_fn(batch):
    """
    Return as outputs the compositeImage+trimap and the alphaMask
    """
    images = []
    masks = []
    
    for (image, trimap, mask) in batch:
        mask = mask.unsqueeze(0)
        trimap = trimap.unsqueeze(0)
        image = torch.cat([image, trimap], 0).unsqueeze(0)
        
        images.append(image)
        masks.append(mask)

    images = torch.cat(images, 0)
    masks = torch.cat(masks, 0)

    return (images, masks)


_TRAIN_FOREGROUND_DIR_ = "./Dataset/Training_set/CombinedForeground"
_TRAIN_BACKGROUND_DIR_ = "./Dataset/Background/COCO_Images"
_TRAIN_ALPHA_DIR_ = "./Dataset/Training_set/CombinedAlpha"
_NETWORK_INPUT_ = (320,320)
_COMPUTE_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NUM_EPOCHS_ = 50
_BATCH_SIZE_ = 4
_NUM_WORKERS_ = multiprocessing.cpu_count()

tripleTransforms = transforms.Compose([
    RandomTrimapCrop([(320, 320), (480, 480), (640, 640)], probability=0.8),
    Resize(_NETWORK_INPUT_),
    ToTensor()
])

trainingDataset = MattingDataset(
                        _TRAIN_FOREGROUND_DIR_, _TRAIN_BACKGROUND_DIR_, _TRAIN_ALPHA_DIR_, 
                        allTransform=tripleTransforms
                    )
trainDataloader = torch.utils.data.DataLoader(
                            trainingDataset, batch_size=_BATCH_SIZE_, shuffle=True, num_workers=_NUM_WORKERS_, collate_fn=batch_collate_fn)

model = EncoderDecoderNet()
refinementModel = RefinementNet(inputChannels=5)



if __name__ == "__main__":

    # idx = random.randint(0, len(trainingDataset))
    # print(f"using image index = {idx}")
    # compositeImage, trimap, alphaMask = trainingDataset[idx]

    # compositeImage.show()
    # trimap.show()
    # alphaMask.show()
    
    # print(f"Dataset length = {len(trainingDataset)}")

    # x = transform_input((compositeImage, trimap))
    # print(f"Composite Input size = {x.size()}")
    
    # predictedMask = model(x)

    # print(predictedMask)
    # print(f"Prediction size = {predictedMask.size()}")

    # finalMask = refinementModel(x, predictedMask)
    # print(finalMask)
    # print(f"Final mask size = {finalMask.size()}")

    # predictedMask = transforms.ToPILImage()(finalMask[0] * 255)
    # predictedMask.show()


    optimiser = torch.optim.SGD([
                    {'params': model.parameters(), 'lr': 1e-2},
                    {'params': refinementModel.parameters(), 'lr': 1e-2}
                ], momentum=0.9)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        refinementModel = torch.nn.DataParallel(refinementModel)
    model.to(_COMPUTE_DEVICE_)
    refinementModel.to(_COMPUTE_DEVICE_)

    trainStart = time.time()
    avgTrainLoss = []

    for epoch in range(_NUM_EPOCHS_):
        print(f"Epoch {epoch+1}/{_NUM_EPOCHS_}")
        epochLoss = 0.0
        epochStart = time.time()

        model.train()
        refinementModel.train()

        for idx, data in enumerate(trainDataloader, 0):
            with torch.set_grad_enabled(True):
                compositeImages, groundTruthMasks = data
                
                predictedMasks = model(compositeImages)

                refinedMasks = refinementModel(compositeImages,  predictedMasks)

                print(f"Final Prediction size = {refinedMasks.size()}")

                #TODO: Sum up the losses from the model & refinementModel
                predictedMasks = predictedMasks.squeeze(1)
                print(f"Reshaped predicted masks = {predictedMasks.size()}")
                modelLoss = alpha_prediction_loss(predictedMasks, groundTruthMasks)
                exit(0)

