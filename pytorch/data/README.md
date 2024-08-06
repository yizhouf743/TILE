TILE using Cifar10 and Tiny-Imagenet dataset. \\
Tiny-Imagenet dataset can be download at: https://www.image-net.org/ \\
Cifar10 dataset can be download using set download option be True, as shown below (in Cifar10 test code): \\
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)\\
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)\\
