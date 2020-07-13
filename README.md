# Deep-Image-Matting
Predicting an alpha matte from an image and a trimap.


# Dataset
- [x] Download 100 COCO backgrounds. I'd be using COCO backgrounds as they're easier to fetch.

# Network structure
- I'd be using a different encoder - decoder architecture to the paper
- I wont be using a pretrained vgg net
- Used this resource to determine the output size of the network http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html and https://pytorch.org/docs/stable/nn.html#convtranspose2d

# Inference on Real World images
- [ ] Try using a segmentation algorithm and random dilation to build a trimap
    - [ ] This can be used to generate alpha mattes for real world images
