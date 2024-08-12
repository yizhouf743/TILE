# TILE
This is the project for the work: "TILE: Input Structure Optimization for Neural Networks to Accelerate Secure Inference".
The codes are still under developments, and should not be used in any security sensitive product.

# Content:
pytorch: The implmentaion for TILE on VGG16 and Resnet50 on Cifar10 and Tiny-Imagenet, including TILE searching, fine-tuning, running inference and so on.

EzPC: A framework that help to covert pytorch code to HE-friendly cpp code and test TILE module/network performance. 

The test code for Conv-Relu(the basic unit on vgg and resnet) with TILE on SCI/tests/test_field_TILE.cpp, this is example for the basic function we use for network testing.

For example:

To run the test code, Please complie EzPC first, then try:

./EzPC/SCI/build/bin/\<test\> r=1 [port=port] & ./EzPC/SCI/build/bin/\<test\> r=2 [port=port]

The code for end-to-end network testing will be update soon.

# Notes:
We are using model pruning technologies from the paper: "MOSAIC: A Prune-and-Assemble Approach for Efficient Model Pruning in Privacy-Preserving Deep Learning". 
We are not showing the code related to this paper, as it is not publicly available. We will update this section once their code becomes publicly accessible.

The implementation to reproduce the results in the paper can be found here: EzPC/SCI/.

We will optimize and cleanup the code in this repo.

The code current updated on: https://anonymous.4open.science/r/TILE-88D2

