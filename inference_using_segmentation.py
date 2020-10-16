import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
from Dataset.MattingDataset import MattingDataset
from dataset_transforms import Resize, ToTensor
from PIL import Image, ImageFilter, ImageChops
from model import EncoderDecoderNet, RefinementNet

def deeplabV3():
    # https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    return models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    
def test_image():
    img = Image.open("./demo/Julia.jpeg")
    sz = img.size
    newSize = (sz[0]//5, sz[1]//5)
    img = img.resize(newSize, Image.BICUBIC)
    img.show()
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

def create_trimap(alphaMask):
    segmentedAlpha = np.array(alphaMask)
    segmentedAlpha[segmentedAlpha > 0] = 255
    segmentedAlpha = Image.fromarray(segmentedAlpha)
    
    dilatedMask = segmentedAlpha.filter(ImageFilter.MaxFilter(9))
    # middle step: threshold the dilatedMask to 127
    dilatedMask = np.array(dilatedMask)
    dilatedMask[dilatedMask > 127] = 127
    dilatedMask = Image.fromarray(dilatedMask)
    trimap = ImageChops.add(segmentedAlpha, dilatedMask)
    trimap = trimap.convert("L")

    return trimap

_TEST_FOREGROUND_DIR_ = "./Dataset/Test_set/Adobe_licensed_images/fg"
_TEST_BACKGROUND_DIR_ = "./Dataset/Background/COCO_Images"
_TEST_ALPHA_DIR_ = "./Dataset/Test_set/Adobe_licensed_images/alpha"
_TEST_TRIMAP_DIR_ = "./Dataset/Test_set/Adobe_licensed_images/trimaps"
_NETWORK_INPUT_ = (320,320)

tf = transforms.Compose([
    Resize(_NETWORK_INPUT_),
    ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testDataset = MattingDataset(_TEST_FOREGROUND_DIR_, _TEST_BACKGROUND_DIR_, _TEST_ALPHA_DIR_,
                        trimapDir=_TEST_TRIMAP_DIR_, allTransform=tf, imageTransforms=None
                    )
model = EncoderDecoderNet()
refinementModel = RefinementNet(inputChannels=5)


if __name__ == "__main__":
    segModel = deeplabV3()
    segModel.eval()

    # Using test image taken by myself
    img = test_image()

    # Get the trimap from the segmentation model
    print(f"Computing image segmentations")
    segmentations = segModel(img)['out'][0]
    predSegmentations = segmentations.argmax(0)
    
    # convert the segmentation to a black and white class
    # predictions for humans are labelled 15
    with torch.no_grad():
        predSegmentations = predSegmentations.int()
        placeholder = torch.zeros_like(predSegmentations)
        placeholder[predSegmentations == 15] = 255
        segmentedImage = transforms.ToPILImage()(placeholder)
        segmentedImage = segmentedImage.convert("RGB")
        segmentedImage.show()
        segmentedImage.save("./demo/Julia_Segmented.jpeg", "jpeg")

        print(f"Creating trimap from the segmented image")
        trimap = create_trimap(segmentedImage)
        trimap.show()
        trimap.save("./demo/Julia_Trimap.jpeg", "jpeg")
        trimap = transforms.ToTensor()(trimap).unsqueeze(0)

        X = torch.cat([img, trimap], 1)
        print(X.size())
        imgSize = X.size()[2:]
        
        # load the encoder-decoder & refinement model
        print(f"Loading pretrained encoder-decoder & refinement model")
        model.load_state_dict(torch.load("./model.pth"))
        refinementModel.load_state_dict(torch.load("./refinement_model.pth"))

        print(f"Computing alpha mask")
        predMask = model(X)
        # As I'm using an arbitrary sized input, it is possible the decoder output size is not equal to the
        # encoder input size. Therefore we use `interpolate` to resize it.
        predMask = F.interpolate(predMask, imgSize)
        print(predMask.size())
        print(f"Refining alpha mask")
        refinedMask = refinementModel(X, predMask)
        print(refinedMask.size())
        predMask = predMask.squeeze(1)
        refinedMask = refinedMask.squeeze(1)
        predMask = transforms.ToPILImage()(predMask[0])
        predMask.show()
        predMask.save("./demo/Julia_EncDec_Mask.jpeg", "jpeg")
        masks = transforms.ToPILImage()(refinedMask[0])
        masks.save("./demo/Julia_Refined_Mask.jpeg", "jpeg")
        masks.show()

        trueForeground = img * refinedMask.unsqueeze(1)
        fg = transforms.ToPILImage()(trueForeground[0])
        fg.show()
        fg.save("./demo/Julia_FG.jpeg", "jpeg")