import torch



def alpha_prediction_loss(predAlpha, trueAlpha):
    """
    Both inputs are expected to be in the form BxSxS as this function operates
    on a batch of single channel images with values in the range of 0 - 1
    """
    squareEps = torch.tensor(1e-6).pow(2).float()
    numInBatch = predAlpha.size(0)
    difference = predAlpha - trueAlpha
    squaredDifference = torch.pow(difference, 2) + squareEps
    rootDiff = torch.sqrt(squaredDifference)
    totalLoss = rootDiff.sum() / numInBatch
    return totalLoss