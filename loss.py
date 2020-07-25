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
    def show(xf):
        for idx in range(xf.size(0)):
            f = tf.ToPILImage()(xf[idx])
            f.show()

    eps = torch.tensor(1e-6).float()
    squareEps = torch.tensor(1e-6).pow(2).float()
    trimaps = compositeImage[:,3,:] * 255
    compositeImage = compositeImage[:,0:3,:] #This removes the trimap added to the last dimension
    
    """
    When using only the trimap to calculate the compositional loss
    It seems this confuses the model and causes the model (either the encoder-decoder or the refinement, whichever it is 
    applied on) to output only black images.
    But when using the alpha mask for loss calculations, the model outputs as expected
    """
    # blackMask = torch.zeros_like(trueAlpha)
    # unknownTrueMask = torch.where(trimaps == 127, trueAlpha, blackMask)
    # unknownPredictedMask = torch.where(trimaps == 127, predAlpha, blackMask)
    # unknownTrueForeground = compositeImage * unknownTrueMask.unsqueeze(1)
    # unknownPredictedForeground = compositeImage * unknownPredictedMask.unsqueeze(1)
    # difference = unknownPredictedForeground - unknownTrueForeground
    # squaredDifference = torch.pow(difference, 2) + squareEps
    # rootDiff = torch.sqrt(squaredDifference)
    # sumRootDiff = rootDiff.sum(dim=[2,3])
    # sumTrueUnknownForeground = unknownTrueForeground.sum(dim=[2,3]) + eps
    # totalLoss = sumRootDiff / sumTrueUnknownForeground
    # avgLoss = totalLoss.mean(dim=1).mean() # average over the RGB channels and also across the batch


    trueForeground = compositeImage * trueAlpha.unsqueeze(1)
    predictedForeground = compositeImage * predAlpha.unsqueeze(1)
    difference = predictedForeground - trueForeground
    squaredDifference = torch.pow(difference, 2) + squareEps
    rootDiff = torch.sqrt(squaredDifference)
    sumTrueForeground = trueForeground.sum(dim=[2,3]) + eps
    totalLoss = rootDiff.sum(dim=[2,3]) / sumTrueForeground
    avgLoss = totalLoss.mean().mean() # average over the RGB channels and also across the batch
    return avgLoss