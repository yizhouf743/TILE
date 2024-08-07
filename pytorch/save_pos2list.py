import torch, os
import torch.nn as nn

save_location = './experiment_data/VGG_Cifar10/'
file_name ='Partial_Model_no_loss_-1.07_v4_mask.pth'

pos_list = torch.load(save_location + file_name, map_location=torch.device('cpu'))
# Specify the directory to save the text file
directory = 'output/Position_List/vgg_cifar10'
os.makedirs(directory, exist_ok=True)

# Path to the output text file
file_path = os.path.join(directory, 'tensors.txt')

for i, tensor in enumerate(pos_list):
    # Path to the output text file
    file_path = os.path.join(directory, f'tensor_{i}.txt')
    
    # Open the text file in write mode
    with open(file_path, 'w') as f:
        # Convert the tensor to a list and write to the file
        f.write(str(tensor.tolist()))