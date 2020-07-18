import torch
import torchvision.transforms as tf


def alpha_prediction_loss(predAlpha, trueAlpha):
    """
    Both inputs are expected to be in the form BxSxS as this function operates
    on a batch of single channel images with values in the range of 0 - 1
    """
    squareEps = torch.tensor(1e-6).pow(2).float()
    difference = predAlpha - trueAlpha
    
    squaredDifference = torch.pow(difference, 2) + squareEps
    
    rootDiff = torch.sqrt(squaredDifference)
    totalLoss = rootDiff.sum(dim=[1,2]).mean()

    return totalLoss


def compositional_loss(predAlpha, trueAlpha, compositeImage):

    def show_alpha(xf):
        for idx in range(xf.size(0)):
            f = tf.ToPILImage()(xf[idx])
            f.show()

    squareEps = torch.tensor(1e-6).pow(2).float()
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