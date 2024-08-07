from typing import Union, List, Dict, Any, cast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from multiprocessing import freeze_support
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import os, copy, sys
# from model.vgg_imagenet import vgg16
import model.vgg_cifar10_recon as vgg
from torchinfo import summary
from utils import progress_bar
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from layer_importance import hspg
from torchinfo import summary
import torch.nn.functional as F
import torchvision

def train_model_with_purn(net, epoch, optimizer, criterion, scheduler, key):
    start_epoch = 0
    best_acc = 0.0
    
    for epoch in range(start_epoch, epoch):
        freeze_support()
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad() 
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # _, predicted = outputs.max(1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # rewrite model_parameters:
            for i in range(len(key)):
              net.state_dict()[key[i]] = net.state_dict()[key[i]] * mask[key[i]]
            
        train_acc, train_loss = 100. * correct / total, train_loss / (batch_idx + 1)
        best_acc = max(validation(net, criterion), best_acc)
        train_acc, train_loss = 100. * correct / total, train_loss / (batch_idx + 1)    
        scheduler.step()
#    progress_bar(batch_idx, len(dataloaders['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('val acc after retrain', best_acc, '%')
    return best_acc


def dist_loss(t, s):
    ## KD temperature: Dafult set is 1 in paper, but 4 in NNI code:
    T = 0.0001
    prob_t = F.softmax(t/T, dim=1)
    log_prob_s = F.log_softmax(s/T, dim=1)
    dist_loss = F.kl_div(log_prob_s, prob_t, size_average=False) * (T**2) / s.shape[0]
    return dist_loss

def train_model_with_purn_KD(net, baseline_net, epoch, optimizer, criterion, scheduler, key, with_KD, with_prun):
    start_epoch = 0
    best_acc = 0.0
    baseline_net.eval()
    
    for epoch in range(start_epoch, epoch):
        freeze_support()
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad() 
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            if with_KD is True:
              baseline_outputs = baseline_net(inputs)
              loss_kd = dist_loss(baseline_outputs, outputs)
              loss += loss_kd
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # _, predicted = outputs.max(1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # rewrite model_parameters:
            if with_prun is True:
              for i in range(len(key)):
                if ('conv' or 'weight') in key[i]:
                  net.state_dict()[key[i]] = net.state_dict()[key[i]] * mask[key[i]]
            
        train_acc, train_loss = 100. * correct / total, train_loss / (batch_idx + 1)
        best_acc = max(validation(net, criterion), best_acc)
        train_acc, train_loss = 100. * correct / total, train_loss / (batch_idx + 1)    
        scheduler.step()
    print('val acc after retrain', best_acc, '%')
    return best_acc

def validation(net, criterion):
    net.eval()
    #  source.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloaders['val']):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
            
    progress_bar(batch_idx, len(dataloaders['val']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return acc


def search_net(initial_net, baseline_net, epoch, threshold=0.03, pruned=False):
    ## accumulative gradient in one epoch:
    test_loss = 0
    correct = 0
    total = 0
    
    lmbda = 1e-3
    key = list(initial_net.state_dict().keys())
    net = copy.deepcopy(initial_net)
    criterion = nn.CrossEntropyLoss()   
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), 0.01, momentum=0.9, weight_decay=5e-4)    
    decays = [int(epoch * 0.5), int(epoch * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=[int(epoch * 0.5), int(epoch * 0.75)], gamma=0.1) 
    accumulate_step = 1
    start_epoch = 0
    inital_acc = 0
    test = None
    previous_group_sparsity = {}
    grad_dict = {} 
    partial = {}
    acculative_steps = epoch
    test = None    
    
    for epoch in range(start_epoch, epoch):   
      for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
          weight = {k:v.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
          inputs, targets = inputs.to(device), targets.to(device)  
          outputs = net(inputs)
          loss = criterion(outputs, targets)  
          optimizer.zero_grad()  
          loss.backward()
          grad = {k:v.grad.data for k,v in net.named_parameters()}
          new_list = 0
          # rewrite model_parameters:
          if pruned is True:
            for i in range(len(key)):
              net.state_dict()[key[i]] = net.state_dict()[key[i]] * mask[key[i]]
              if key[i] in grad:
                net.state_dict()[key[i]].grad = grad[key[i]] * mask[key[i]]     

          if batch_idx == 0:      
            grad_dict = {k:v.grad.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
          else:
            for k,v in net.named_parameters():
              grad_dict[k] += v.grad.data.view(v.data.shape[0], -1) / (batch_idx + 1)

         
          epsilon = 1
          lmbda = adjust_lambda(lmbda, epoch, decays)
          show_result = False
          
          if batch_idx == len(dataloaders['train']) - 1:
            print('group sparsity: ', previous_group_sparsity)
            print('Show intermediate result on Epoch: ', epoch)
            show_result = True
          
          grad_dict = {k:v.grad.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
          new_list = hspg(net, baseline_net, grad_dict, lmbda, epsilon, optimizer, threshold, previous_group_sparsity, partial, show_result)
            
          probs = torch.nn.functional.softmax(outputs, dim=1)
          pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(targets.view_as(pred)).sum().item()
          total += targets.size(0)     

          acc = 100.* correct/total
          test_loss += loss.item()
          test_loss = test_loss/(batch_idx+1)
          
          if (test != new_list) and (test):
            print(batch_idx, test, new_list)
            print('search algorithm break')
            sys.exit("Error message")
          else:
            test = new_list
          
          if batch_idx == len(dataloaders['train']) - 1: 
            print('new network list: ', test)
            print('current val acc: ', 100. * correct / total, '%', ' in epoch: #', epoch)
            print('val loss: ', test_loss)

    scheduler.step()
    hspg_score = list(previous_group_sparsity.values())
    hspg_score = torch.as_tensor(hspg_score) /(batch_idx+1)
    
    for name in list(partial.keys()):
        partial[name] = partial[name] /(batch_idx+1)
    print('average hspg score: ')
    print(hspg_score)
    return hspg_score, partial

def cal_partial_order(replace_order, search_range, cn=[2, 2, 2], threshold=2):
    partial_key = list(replace_order.keys())
    chans_per_cipher = [4, 16, 64]
    partial_order = []
    remove_item = []
    counter = 0

    for name in partial_key:  
      if counter < 3:
        c_n, cpc = cn[0], chans_per_cipher[0]
      elif counter in list(range(3, 9)):
        c_n, cpc = cn[1], chans_per_cipher[1]
      else:
        c_n, cpc = cn[2], chans_per_cipher[2]

      layer_order = replace_order[name]
      layer_order = torch.log10(layer_order)
      indices = torch.where(layer_order < threshold)[0]
      #  Show top 3 HSPG score for each layer:
      print(name, torch.topk(torch.unique(layer_order), 3))
      # # Check Prunned Position & remove overlap part:
      layer_weight = pre_trained_model[name]
      score = layer_weight.abs().sum(dim=(2, 3))
      score[score > 0] = 1
      count_np_col = torch.count_nonzero(score, dim=0)
      prunned_input_chans_pos = torch.where(count_np_col == 0)[0]
      non_prunned_chans = torch.where(count_np_col != 0)[0].numel()
      # print(name, indices.shape)
      if prunned_input_chans_pos.numel() > 0:
        indices = indices[~torch.isin(indices, prunned_input_chans_pos)]
      # # 取整：
      if indices.numel() < non_prunned_chans - cpc:
        if indices.numel() < cpc:
          if search_range[counter] not in remove_item:
            indices = torch.tensor([])
            remove_item.append(search_range[counter])
            
      if indices.numel() != non_prunned_chans:
        ar = torch.floor_divide(indices.numel(), cpc) * cpc
        indices = indices[:int(ar.item())]

      partial_order.append(indices)
      counter += 1
      print('number of apply channel: ', name, cpc)
      print('apply ratio for ', name, torch.numel(indices), ' / ', non_prunned_chans,' = ', 100 * torch.numel(indices) / non_prunned_chans, '%')
      print('ciphertext for apply: ',  torch.numel(indices) / cpc,  " / ", torch.ceil(torch.tensor(non_prunned_chans / cpc)).item())
    [search_range.remove(x) for x in remove_item]
    # print('remove layer ', remove_item, ' from test list')
    return partial_order, search_range
    
def initial_fine_tunning(search_result, train_epoch):
    new_key = list(search_result.state_dict().keys())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, search_result.parameters()), 0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    best_acc = train_model_with_purn(search_result, train_epoch, optimizer, criterion, scheduler, new_key)
    return best_acc
    
def fine_tunning(search_result, train_epoch, baseline_net=None, with_KD=False, with_prun=True):
    new_key = list(search_result.state_dict().keys())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, search_result.parameters()), 0.005, momentum=0.9, weight_decay=5e-4)
#    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[int(train_epoch * 0.5), int(train_epoch * 0.75)], gamma=0.1)
    best_acc = train_model_with_purn_KD(search_result, baseline_net, train_epoch, optimizer, criterion, scheduler, new_key, with_KD, with_prun)
    return best_acc
    
def adjust_lambda(lmbda, epoch, decays):
    next_lmbda = 0.0
    if epoch in decays:
      next_lmbda = 0.0 if lmbda <= 1e-6 else lmbda / 10.0
      lmbda = next_lmbda
    return lmbda
    
    
def cal_importance(net):
   pre_trained_model = net.state_dict()
   key = list(pre_trained_model.keys())
   partial_order = []
   
   for name in key:
    if all([x in name for x in ['features', 'conv', 'weight']]):
      layer_weight = pre_trained_model[name]
      indices = torch.tensor(list(range(layer_weight.shape[1])))
      partial_order.append(indices)
   return partial_order  


def approximate_transfer(net):
  pre_trained_model = net.state_dict()
  key = list(pre_trained_model.keys())
  counter = 0
  tile_type = []

  for name in key:
    if all([x in name for x in ['features', 'conv', 'weight']]):
      if counter < 4:
        c_n = 4
      elif counter in list(range(4, 10)):
        c_n = 16
      else:
        c_n = 64
      layer_weight = pre_trained_model[name]
      score = layer_weight.abs().sum(dim=(2, 3))
      score[score > 0] = 1

      if 'features.0' not in name:
        count_np_col = torch.count_nonzero(score, dim=0).numel()
        count_np_row = torch.count_nonzero(score, dim=1).numel()
        print('transfered shape for ', name, ' is:',  count_np_row, "x", count_np_col)
        left_side = 2 * count_np_col * 8
        right_side = 0.6 * count_np_row *(c_n-1)
        if left_side > right_side:
            print(name, 'apply internal tile')
            tile_type.extend([0])
        else:
           print(name, 'apply external tile')
           tile_type.extend([1])
      counter += 1
  return tile_type    
            
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

# Data Loader:
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
dataloaders = {}                  
dataloaders['train'], dataloaders['val'] = trainloader, testloader

ps = [2, 2, 2, 2, 'M', 2, 2, 2, 2, 2, 2, 'M', 2, 2, 2,'M']
search_range = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14]
tile_type = ['N', 0, 0, 0, 'N', 1, 0, 0, 1, 0, 0, 'N', 1, 1, 1,'N']

threshold = 0.027 
save_location = './experiment_data/VGG_Cifar10/'
print('the file will save on: ', save_location)

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
     # "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "D": [64, 64, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

if __name__ == '__main__':
    # purnned VGG-16 "D": [64, 64, 128, 128, "M", 256, 256, 256, 512, 512, 512, "M", 512, 512, 512, "M"]
    pre_trained_model = torch.load(save_location + 'baseline_model.pth', map_location=torch.device(device))
    new_key = list(pre_trained_model.keys())
    ## update the search range to avoid public layers:
    train_epoch = 60
    
    guilde_net = vgg.VGG(vgg.make_layers(cfgs["D"], select_range=None, ps=ps))
    guilde_para = torch.load(save_location + 'baseline_model.pth', map_location=torch.device(device))
    # guilde_para = torch.load('baseline_model.pth', map_location=torch.device(device))
    guilde_net.load_state_dict(guilde_para, strict=True)
    guilde_net = guilde_net.to(device)
    criterion = nn.CrossEntropyLoss()  
    guilde_acc = validation(guilde_net, criterion)
    print('teacher acc: ',  guilde_acc, '%')
    approximate_transfer(guilde_net)

    acc = 0
    # find the best threshold = -0.5
    thres = -1.06
    # # Partial_Apply:
    test = search_range
    threshold = 1
    counter = 1
    partial_order = cal_importance(guilde_net)
    pre_trained_model = torch.load(save_location + 'baseline_model.pth', map_location=torch.device(device))
    net = vgg.VGG(vgg.make_layers(cfgs["D"], select_range=test, ps=ps, tile_type=tile_type, partial=partial_order))
    net.load_state_dict(pre_trained_model, strict=True)   
    net = net.to(device)  
    
    hspg_score, replace_order = search_net(net, guilde_net, 1, threshold, pruned=False)
    torch.save(replace_order, save_location + 'inital_mask_prun.pth')

    replace_order = torch.load(save_location + 'inital_mask_prun.pth', map_location=torch.device(device)) 
    # print('finish generate HSPG replace order for Prunned Model')
    last_order = None

    while (acc < guilde_acc) and (thres >= -2):
      search_target = search_range
      partial_order, test = cal_partial_order(replace_order, search_target, threshold=thres)

      skip_comment = True
      if last_order is not None:
        for i in range(len(partial_order)):
          if torch.numel(partial_order[i]) != torch.numel(last_order[i]):
              skip_comment = False
        if skip_comment == True:
            print("skip threshold: ", thres)
            thres = thres - 0.01
            continue

      print('and pre-defined threshold is less than ', thres)
      print('start fine-tunning for: ', test)
      net = vgg.VGG(vgg.make_layers(cfgs["D"], select_range=test, ps=ps, tile_type=tile_type, partial=partial_order))
      pre_trained_model = torch.load(save_location + 'baseline_model.pth', map_location=torch.device(device)) 
      net = net.to(device) 
      net.load_state_dict(pre_trained_model, strict=True)
      acc = fine_tunning(net, train_epoch, guilde_net, with_KD=True, with_prun=False)  

      if acc >= guilde_acc:
        torch.save(partial_order, save_location + 'Partial_Model_no_loss_' + str(thres) +'_v4_mask.pth')
        torch.save(net.state_dict(), save_location + 'Partial_Model_no_loss_' + str(thres) +'_v4.pth')
        print('save parameter path: ', save_location + 'Partial_Model_no_loss_' + str(thres) +'_v4.pth')
        sys.exit("find best combanation")
        
      if thres > 1:
        thres = thres / 2
      else:
        thres = thres - 0.01
      last_order = partial_order
      del net, test, search_target
