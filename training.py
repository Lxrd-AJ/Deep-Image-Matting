import torch
import torchvision.transforms as transforms
import random
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt
from PIL import Image
from Dataset.MattingDataset import MattingDataset
from model import EncoderDecoderNet, RefinementNet
from dataset_transforms import RandomTrimapCrop, Resize, ToTensor
from loss import alpha_prediction_loss, compositional_loss




def clip_gradients(models):
    for m in models:
        torch.nn.utils.clip_grad_norm_(m.parameters(), _GRADIENT_CLIP_)


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
_NUM_EPOCHS_ = 30 #200 #TODO: Remove
_BATCH_SIZE_ = 8 #TODO: Increase this if using a GPU
_NUM_WORKERS_ = multiprocessing.cpu_count()
_LOSS_WEIGHT_ = 0.4 #0.5
_GRADIENT_CLIP_ = 2.5

tripleTransforms = transforms.Compose([
    RandomTrimapCrop([(320, 320), (480, 480), (640, 640)], probability=0.7),
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

                predictedMasks = predictedMasks.squeeze(1)
                refinedMasks = refinedMasks.squeeze(1)
                
                modelAlphaLoss = alpha_prediction_loss(predictedMasks, groundTruthMasks)                
                refinedAlphaLoss = alpha_prediction_loss(refinedMasks, groundTruthMasks)
                lossAlpha = modelAlphaLoss + refinedAlphaLoss
                lossComposition = compositional_loss(predictedMasks, groundTruthMasks, compositeImages)
                totalLoss = _LOSS_WEIGHT_ * lossAlpha + (1 - _LOSS_WEIGHT_) * lossComposition
                epochLoss += totalLoss.item()

                if idx % 100 == 0:
                    print(f"\tIteration {idx+1}/{len(trainingDataset)}")
                    print("-----" * 15)
                    print(f"\t Encoder-Decoder alpha loss = {modelAlphaLoss}")
                    print(f"\t Refinement model alpha loss = {refinedAlphaLoss}")
                    print(f"\t Alpha loss = {lossAlpha}")
                    print(f"\t Composition loss = {lossComposition}")
                    print(f"\t Total Loss = {totalLoss}")
                    print()

                optimiser.zero_grad()
                totalLoss.backward()

                #NB: `lossAlpha` can be high at first e.g 107,001. This can cause high gradient updates and can
                #   make training unstable. The gradients might need to be clipped to help training.
                #   The model still trains nicely without clipping, this just gives you that smooth loss function
                clip_gradients([model, refinementModel])

                optimiser.step()


        epochLoss = epochLoss / len(trainDataloader)
        epochElapsed = time.time() - epochStart
        print(f"\t Average Train Epoch loss is {epochLoss:.2f} [{epochElapsed//60:.0f}m {epochElapsed%60:.0f}s]")
        print("-----" * 15)
        avgTrainLoss.append(epochLoss)

        plt.plot(avgTrainLoss, 'r', label='Train')
        plt.xticks(np.arange(0,_NUM_EPOCHS_+10,10))
        plt.title(f"Training loss using a dataset of {len(trainingDataset)} images")
        plt.savefig(f"TrainLoss{len(trainingDataset)}Items.png")

    #Make a sample prediction
    idx = random.choice(range(0, len(trainingDataset)))
    img_, trimap, gMasks = trainingDataset[0]
    trimap = trimap.unsqueeze(0)
    gMasks = gMasks.unsqueeze(0)
    img = torch.cat([img_, trimap], 0).unsqueeze(0)

    masks = model(img)
    x  = transforms.ToPILImage()(img_[0])
    x.show()
    x = transforms.ToPILImage()(gMasks[0] * 255)
    x.show()
    x = transforms.ToPILImage()(trimap[0] * 255)
    x.show()
    x = transforms.ToPILImage()(masks[0] * 255)
    x.show()
    masks = refinementModel(img, masks)
    x = transforms.ToPILImage()(masks[0] * 255)
    x.show()

    cImg = img_ * masks.squeeze(0)
    x = transforms.ToPILImage()(cImg[0])
    x.show()

