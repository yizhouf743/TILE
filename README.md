# TILE
This repository for the work: "TILE: Input Structure Optimization for Neural Networks to Accelerate Secure Inference". The corresponding paper has been accepted by ACSAC24. 
The codes are still under cleanup and development, and should not be used in any security-sensitive product.

# Content:
PyTorch: The implementation for TILE on VGG16 and Resnet50 on Cifar10 and Tiny-Imagenet, including TILE searching, fine-tuning, running inference, and so on.

EzPC: A framework that helps to covert pytorch code to HE-friendly cpp code and test TILE module/network performance. 

Currently, only experiment scripts for the plaintext model and ciphertext test demo for Conv-Relu(the basic unit on vgg and resnet) are available on SCI/tests/test_field_TILE.cpp. 

To run the ciphertext test demo, Please compile EzPC/SCI first, then try:

./EzPC/SCI/build/bin/TILE-HE r=1 & ./EzPC/SCI/build/bin/TILE-HE r=2.

# Notes:
We are using model pruning technologies from the paper: "MOSAIC: A Prune-and-Assemble Approach for Efficient Model Pruning in Privacy-Preserving Deep Learning". 
We are not showing the code related to this paper, as it is not publicly available. We will update this section once their code becomes publicly accessible.

The implementation to reproduce the results in the paper can be found here: EzPC/SCI/.

We will optimize and clean up the code in this repo.

# Network setting on Linux:
Run sudo ./throttle.sh lan to simulate the communication network in a LAN environment on one machine. We provide three network setting options: lan/wan/mobile_US_avg for lab environments.
