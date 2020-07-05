# Deep-Image-Matting
Predicting an alpha matte from an image and a trimap.


# Dataset
- [x] Download 100 COCO backgrounds. I'd be using COCO backgrounds as they're easier to fetch.

# Network structure
- [ ] Define Encoder-Decoder net
    - [ ] To support pretrained vgg or not?
- [ ] Define refinement net

# Inference on Real World images
- [ ] Try using a segmentation algorithm and random dilation to build a trimap
    - [ ] This can be used to generate alpha mattes for real world images
