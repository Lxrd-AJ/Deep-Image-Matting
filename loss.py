import torch
import torchvision.transforms as tf


def alpha_prediction_loss(predAlpha, trueAlpha):
    """
    Both inputs are expected to be in the form BxSxS as this function operates
    on a batch of single channel images with values in the range of 0 - 1
    """
    eps = torch.tensor(1e-6).float()
    squareEps = eps.pow(2)
    difference = predAlpha - trueAlpha
    
    squaredDifference = torch.pow(difference, 2) + squareEps
    
    rootDiff = torch.sqrt(squaredDifference)
    sumRootDiff = rootDiff.sum(dim=[1,2])
    sumTrueAlpha = trueAlpha.sum(dim=[1,2]) + eps
    totalLoss = sumRootDiff / sumTrueAlpha
    avgTotalLoss = totalLoss.mean()

    return avgTotalLoss


def compositional_loss(predAlpha, trueAlpha, compositeImage):

    def show_alpha(xf):
        for idx in range(xf.size(0)):
            f = tf.ToPILImage()(xf[idx])
            f.show()

    squareEps = torch.tensor(1e-6).pow(2).float()
    trimaps = compositeImage[:,3,:] * 255
    print(trimaps[(trimaps > 127)].mean())
    print(trimaps[(trimaps < 127)].mean())
    #TODO NB: there seems to be values other than 0, 127 & 255 in the trimap
    unknownIndices = (trimaps == 127).nonzero()
    unknownPixels = trimaps[trimaps == 127].nonzero()
    print(unknownIndices.size())
    print(trimaps[unknownIndices].size())
    print(unknownPixels)
    print(unknownPixels.size())
    exit(0)
    compositeImage = compositeImage[:,0:3,:] #This removes the trimap added to the last dimension
    trueForeground = compositeImage * trueAlpha.unsqueeze(1)
    # show_alpha(trueForeground)
    predictedForeground = compositeImage * predAlpha.unsqueeze(1)
    # show_alpha(predictedForeground)
    difference = predictedForeground - trueForeground
    squaredDifference = torch.pow(difference, 2) + squareEps
    rootDiff = torch.sqrt(squaredDifference)
    totalLoss = rootDiff.sum(dim=[2,3]).mean().mean() # average over the RGB channels and also across the batch
    
    return totalLoss