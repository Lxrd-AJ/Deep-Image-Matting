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
from dataset_transforms import RandomTrimapCrop, Resize, ToTensor, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, RandomBlur, RandomAffine
from loss import alpha_prediction_loss, compositional_loss, sum_absolute_difference, mean_squared_error



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

def evaluate(model, refinementModel, testDataloader):
    model.eval()
    refinementModel.eval()
    evalLoss = 0.0
    with torch.no_grad():
        for idx, data in enumerate(trainDataloader, 0):
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
            evalLoss += totalLoss
    return evalLoss / len(testDataloader)


_TRAIN_FOREGROUND_DIR_ = "./Dataset/Training_set/CombinedForeground"
_TRAIN_BACKGROUND_DIR_ = "./Dataset/Background/COCO_Images"
_TRAIN_ALPHA_DIR_ = "./Dataset/Training_set/CombinedAlpha"
_TEST_FOREGROUND_DIR_ = "./Dataset/Test_set/Adobe_licensed_images/fg"
_TEST_BACKGROUND_DIR_ = "./Dataset/Background/COCO_Images"
_TEST_ALPHA_DIR_ = "./Dataset/Test_set/Adobe_licensed_images/alpha"
_TEST_TRIMAP_DIR_ = "./Dataset/Test_set/Adobe_licensed_images/trimaps"

_NETWORK_INPUT_ = (320,320)
_COMPUTE_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NUM_EPOCHS_ = 45 #200 #TODO: Remove 50
_BATCH_SIZE_ = 8 #TODO: Increase this if using a GPU
_NUM_WORKERS_ = multiprocessing.cpu_count()
_LOSS_WEIGHT_ = 0.6
_GRADIENT_CLIP_ = 10

tripleTransforms = transforms.Compose([
    RandomRotation(probability=0.5, angle=180),
    RandomVerticalFlip(probability=0.5),
    RandomHorizontalFlip(probability=0.5),
    RandomTrimapCrop([(320, 320), (480, 480), (640, 640)], probability=0.7),
    Resize(_NETWORK_INPUT_),
    ToTensor()
])

imageTransforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    transforms.RandomGrayscale(p=0.3),
    RandomBlur(probability=0.1)
])

trainingDataset = MattingDataset(
                        _TRAIN_FOREGROUND_DIR_, _TRAIN_BACKGROUND_DIR_, _TRAIN_ALPHA_DIR_, 
                        allTransform=tripleTransforms, imageTransforms=None
                    )
testDataset = MattingDataset(_TEST_FOREGROUND_DIR_, _TEST_BACKGROUND_DIR_, _TEST_ALPHA_DIR_,
                        trimapDir=_TEST_TRIMAP_DIR_, allTransform=transforms.Compose([Resize(_NETWORK_INPUT_),ToTensor()]), imageTransforms=None
                    )

trainDataloader = torch.utils.data.DataLoader(
                            trainingDataset, batch_size=_BATCH_SIZE_, shuffle=True, num_workers=_NUM_WORKERS_, collate_fn=batch_collate_fn)
testDataloader = torch.utils.data.DataLoader(
                            testDataset, batch_size=_BATCH_SIZE_, shuffle=True, num_workers=_NUM_WORKERS_, collate_fn=batch_collate_fn)

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
    avgTestLoss = []

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
                with torch.no_grad():
                    sad = sum_absolute_difference(groundTruthMasks, refinedMasks)
                    mse = mean_squared_error(groundTruthMasks, refinedMasks, compositeImages)

                if idx % 100 == 0:
                    print(f"\tIteration {idx+1}/{len(trainingDataset)}")
                    print("-----" * 15)
                    print(f"\t Encoder-Decoder alpha loss = {modelAlphaLoss}")
                    print(f"\t Refinement model alpha loss = {refinedAlphaLoss}")
                    print(f"\t Alpha loss = {lossAlpha}")
                    print(f"\t Composition loss = {lossComposition}")
                    print(f"\t Total Loss = {totalLoss}")
                    print(f"\t {'***' * 5}")
                    print(f"\t Metrics:")
                    print(f"\t {'***' * 5}")
                    print(f"\t Sum absolute difference: {sad}")
                    print(f"\t Mean Squared Error: {mse}")
                    print()

                optimiser.zero_grad()
                totalLoss.backward()

                # Gradient clipping doesn't really make that much of a difference
                clip_gradients([model, refinementModel])

                optimiser.step()


        epochLoss = epochLoss / len(trainDataloader)
        epochElapsed = time.time() - epochStart
        print(f"\t Average Train Epoch loss is {epochLoss:.2f} [{epochElapsed//60:.0f}m {epochElapsed%60:.0f}s]")
        # Evaluate on the test set
        epochTestLoss = evaluate(model, refinementModel, testDataloader)
        print(f"\t Average Test Epoch loss is {epochTestLoss:.2f}")
        print("-----" * 15)
        avgTrainLoss.append(epochLoss)
        avgTestLoss.append(epochTestLoss)

        plt.plot(avgTrainLoss, 'r', label='Train')
        plt.plot(avgTestLoss, 'b', label="test")
        plt.xticks(np.arange(0,_NUM_EPOCHS_+10,10))
        plt.title(f"Training & Test loss using a dataset of {len(trainingDataset)} images")
        plt.savefig(f"TrainTestLoss{len(trainingDataset)}Items.png")
        

    trainingElapsed = time.time() - trainStart
    print(f"\nTotal training time is {trainingElapsed//60:.0f}m {trainingElapsed%60:.0f}s")

    # save the models to disk
    torch.save(model.state_dict(), "./model.pth")
    torch.save(refinementModel.state_dict(), "./refinement_model.pth")
    
    #Make a sample prediction
    with torch.no_grad():
        idx = random.choice(range(0, len(testDataset)))
        img_, trimap, gMasks = testDataset[idx]
        trimap = trimap.unsqueeze(0)
        gMasks = gMasks.unsqueeze(0)
        img = torch.cat([img_, trimap], 0).unsqueeze(0)

        x  = transforms.ToPILImage()(img_)
        x.show()
        
        # x = transforms.ToPILImage()(gMasks[0] * 255)
        # x.show()

        masks = model(img)
        x = transforms.ToPILImage()(masks[0])
        x.show()
        cImg = img_ * masks.squeeze(0)
        x = transforms.ToPILImage()(cImg)
        x.show()


        masks = refinementModel(img, masks)
        x = transforms.ToPILImage()(masks[0])
        x.show()
        cImg = img_ * masks.squeeze(0)
        x = transforms.ToPILImage()(cImg)
        x.show()
