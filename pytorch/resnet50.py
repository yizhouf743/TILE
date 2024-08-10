import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from multiprocessing import freeze_support
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import os, copy, sys
import model.resnet50_tiny_recon as res
import torch.nn.functional as F
from torchinfo import summary
from utils import progress_bar
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from layer_importance_v3 import hspg_resnet50
from torchinfo import summary
import numpy as np


def dist_loss(t, s):
    T = 4
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
        progress_bar(batch_idx, len(dataloaders['train']), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        train_acc, train_loss = 100. * correct / total, train_loss / (batch_idx + 1)
        best_acc = max(validation(net, criterion), best_acc)
        train_acc, train_loss = 100. * correct / total, train_loss / (batch_idx + 1)    
        scheduler.step()
    print('val acc after retrain', best_acc, '%')
    return best_acc

def validation(net, criterion, data_set='val'):
    net.eval()
    #  source.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloaders[data_set]):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
            
    progress_bar(batch_idx, len(dataloaders['val']), 'Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
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
    partial = {}
    grad_dict = {} 
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
            print('Show intermediate result on Epoch: ', epoch)
            show_result = True
          
          grad_dict = {k:v.grad.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
          new_list = hspg_resnet50(net, baseline_net, grad_dict, lmbda, epsilon, optimizer, threshold, previous_group_sparsity, partial, show_result)
            
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
    
def fine_tunning(search_result, train_epoch, baseline_net=None, with_KD=False, with_prun=True):
    new_key = list(search_result.state_dict().keys())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, search_result.parameters()), 0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(train_epoch * 0.5), int(train_epoch * 0.75)], gamma=0.1)
    best_acc = train_model_with_purn_KD(search_result, baseline_net, train_epoch, optimizer, criterion, scheduler, new_key, with_KD, with_prun)
    return best_acc
    
def adjust_lambda(lmbda, epoch, decays):
    next_lmbda = 0.0
    if epoch in decays:
      next_lmbda = 0.0 if lmbda <= 1e-6 else lmbda / 10.0
      lmbda = next_lmbda
    return lmbda
    
             
def count_sparse(net):
    layer_key = list(net.state_dict().keys())
    sparse = {}
    param_dict = {k:v.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
    print('count non-Pruned element rates:')
    
    for name in layer_key:
      if ('conv' or 'weight') in name:
        preds = param_dict[name]
        sparse[name] = torch.count_nonzero(preds)/preds.size().numel()
        print(name, sparse[name] * 100)
        print(preds.shape)
    return sparse


def cal_importance(net):
   pre_trained_model = net.state_dict()
   key = list(pre_trained_model.keys())
   partial_order = []
   
   for name in key:
    if all([x in name for x in ['layer', 'conv', 'weight']]):
      if any([x in name for x in ['conv1', 'conv2', 'conv3']]):
        layer_weight = pre_trained_model[name]
        indices = torch.tensor(list(range(layer_weight.shape[1])))
        partial_order.append(indices)
   return partial_order  

def cal_partial_order(replace_order, search_range, cn=[2, 2, 2, 2], threshold=2):
    partial_key = list(replace_order.keys())
    chans_per_cipher = [2, 8, 32, 128]
    chans_per_cipher2 = [2, 4, 16, 64]
    partial_order = []
    remove_item = []
    counter = 0

    for name in partial_key:  
      if 'conv2' in name:
        if 'layer1' in name:
          cpc = chans_per_cipher2[0]
        elif 'layer2' in name:
          cpc = chans_per_cipher2[1]
        elif 'layer3' in name:
          cpc = chans_per_cipher2[2]
        else:
          cpc = chans_per_cipher2[3]
      else:
        if 'layer1' in name:
          cpc = chans_per_cipher[0]
        elif 'layer2' in name:
          cpc = chans_per_cipher[1]
        elif 'layer3' in name:
          cpc = chans_per_cipher[2]
        else:
          cpc = chans_per_cipher[3]

      layer_order = replace_order[name]
      layer_order = torch.log10(layer_order)
      total_chans = layer_order.numel()

      # # Calculate replace strength based on compression ratio and sprase ratio:  
      layer_weight = pre_trained_model[name]
      score = layer_weight.abs().sum(dim=(2, 3))
      score[score > 0] = 1
      count_np_col = torch.count_nonzero(score, dim=0)
      count_np_row = torch.count_nonzero(score, dim=1)
      rs = torch.count_nonzero(count_np_row).item() / torch.count_nonzero(count_np_col).item()
      sr = torch.count_nonzero(score) / score.numel()

      norm_count = torch.tensor(-(47 - counter))
      sf = torch.exp(0.1 * norm_count)
      # rs = sr * rs * sf
      rs = rs * sf
  
      layer_order = layer_order.view(-1, cpc)
      layer_order = layer_order.sum(dim=1).repeat_interleave(cpc)
      if 'conv2' in name:
        indices = torch.where(layer_order < threshold)[0]
      else:
        # layer_order = layer_order.view(-1, cpc)
        # layer_order = layer_order.sum(dim=1).repeat_interleave(cpc)
        if threshold > 0:
          indices = torch.where(layer_order < (threshold * rs))[0]
        else:
          indices = torch.where(layer_order < (threshold / rs))[0]
        print('inital threshold for', name, 'is', (threshold * rs).item())
      
      # # skip if the number of applied channel cannot fill one ciphertext：
      if indices.numel() < cpc:
        indices = torch.tensor([])
        remove_item.append(search_range[counter])
        partial_order.append(indices)
        counter += 1
        print('number of apply channel: ', name, cpc)
        print('apply ratio for ', name, torch.numel(indices), ' / ', total_chans,' = ', 100 * torch.numel(indices) / total_chans, '%')
        print('ciphertext for apply: ',  torch.numel(indices) / cpc,  " / ", torch.ceil(torch.tensor(total_chans / cpc)).item())
        continue

      if indices.numel() != total_chans:
        ar = torch.floor_divide(indices.numel(), cpc) * cpc
        indices = indices[:int(ar.item())]
          
      partial_order.append(indices)
      counter += 1
      print('number of apply channel: ', name, cpc)
      print('apply ratio for ', name, torch.numel(indices), ' / ', total_chans,' = ', 100 * torch.numel(indices) / total_chans, '%')
      print('ciphertext for apply: ',  torch.numel(indices) / cpc,  " / ", torch.ceil(torch.tensor(total_chans / cpc)).item())
    [search_range.remove(x) for x in remove_item]
    return partial_order, search_range

def approximate_transfer(net):
  pre_trained_model = net.state_dict()
  key = list(pre_trained_model.keys())
  tile_type = []
  chans_per_cipher = [2, 8, 32, 128]
  chans_per_cipher2 = [2, 4, 16, 64]
  counter = 0

  for name in key:
    if all([x in name for x in ['layer', 'conv', 'weight']]):
      if 'conv2' in name:
        if 'layer1' in name:
          c_n = chans_per_cipher2[0]
        elif 'layer2' in name:
          c_n = chans_per_cipher2[1]
        elif 'layer3' in name:
          c_n = chans_per_cipher2[2]
        else:
          c_n = chans_per_cipher2[3]
      else:
        if 'layer1' in name:
          c_n = chans_per_cipher[0]
        elif 'layer2' in name:
          c_n = chans_per_cipher[1]
        elif 'layer3' in name:
          c_n = chans_per_cipher[2]
        else:
          c_n = chans_per_cipher[3]

      layer_weight = pre_trained_model[name]
      score = layer_weight.abs().sum(dim=(2, 3))
      score[score > 0] = 1
      count_np_col = torch.count_nonzero(torch.count_nonzero(score, dim=0)).item()
      count_np_row = torch.count_nonzero(torch.count_nonzero(score, dim=1)).item()
      print('transfered shape for ', name, ' is:',  count_np_row, "x", count_np_col)
      # uncount inp_chans for each out_chans:
      if any([x in name for x in ['conv1', 'conv3']]):
        filter_size = 0
      else:
        filter_size = 9 - 1
      left_side = 2.23 * count_np_col * filter_size
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
data_dir = './data/tiny-imagenet-200/'

data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                ]),

                'val': transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                     std = [ 0.229, 0.224, 0.225 ])
                ])}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                  for x in ['train', 'val']}
dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
                  for x in ['train', 'val']}
                  

replace_point = list(range(48))
replace_pool = replace_point
print('pool size', len(replace_pool))
print(replace_pool)
search_range = replace_pool

## Search Model based on SecureTVM's case:
arg = 'he_friend'

## based on HE-friendly:
save_location = './experiment_data/Resnet50/HE_friendly/'
#  mask = torch.load(save_location+'freeze_start_finetuned_mask.pth', map_location=torch.device(device))
pre_trained_model = torch.load(save_location+'baseline3.0_rename_para.pth', map_location=torch.device(device))
option_save_location = ''
print('the file will save on: ', save_location + option_save_location)


if __name__ == '__main__':
    train_epoch = 60
    layer_key = list(pre_trained_model.keys())

    guilde_net = res.ResNet50(res.Bottleneck,[3,4,6,3])
    guilde_para = torch.load(save_location+'baseline3.0_rename_para.pth', map_location=torch.device(device))
    guilde_net.load_state_dict(guilde_para, strict=True)
    guilde_net = guilde_net.to(device)
    criterion = nn.CrossEntropyLoss()  
    guilde_acc = validation(guilde_net, criterion)
    tile_list = approximate_transfer(guilde_net)
    print('teacher acc: ',  guilde_acc, '%')
    print(tile_list)
    
    # # output partial replace order:
    test = search_range
    # # Initial tile locator:
    partial_order = cal_importance(guilde_net)
    threshold = 3
    version_id = 3
    net = res.ResNet50(res.Bottleneck,[3,4,6,3], replace_point=test, cn=[2, 2, 2, 2], partial=partial_order)
    net.load_state_dict(pre_trained_model, strict=True)     
    net = net.to(device)
    # calculate apply score for each input channel:
    hspg_score, replace_order = search_net(net, guilde_net, 1, threshold)
    torch.save(replace_order, save_location + option_save_location + 'inital_mask_prun.pth')
    print('finish generate apply mask for model')

    replace_order = torch.load(save_location + option_save_location + 'inital_mask_prun.pth', map_location=torch.device(device)) 
    para = pre_trained_model
 
    acc = 0
    step = (1 - 0.8603) / 4
    # thres = 2.8952250000000017
    bottom = 0.7238062500000001
    top = 3.384175 # 1.25
    step = (top - bottom) / 4
    # thres = 3.314325 + step
    thres = top + 4 * step

    ar= [[20, 20, 170, 20], [120, 40, 340, 40], [240, 80, 672, 80], [448, 128, 1344, 128]]
    # last_order = None
    search_target = list(range(48))
    last_order, test = cal_partial_order(replace_order, search_target, threshold=thres)

    while (acc < guilde_acc) and (thres >= -2):
      search_target = list(range(48))
      # define tile position:
      partial_order, test = cal_partial_order(replace_order, search_target, threshold=thres)

      skip_comment = True
      if last_order is not None:
        for i in range(len(partial_order)):
          if torch.numel(partial_order[i]) != torch.numel(last_order[i]):
              skip_comment = False
        if skip_comment == True:
            print("skip threshold: ", thres)
            thres = thres - step
            continue

      print('Output HSPG identifier partial_order when threshold = ', thres)

      net = res.ResNet50(res.Bottleneck,[3,4,6,3], replace_point=test, apply_ratio=ar, partial=partial_order)
      net.load_state_dict(para, strict=True)   
      net = net.to(device) 
      acc = fine_tunning(net, train_epoch, guilde_net, with_KD=True, with_prun=False) 
      if acc >= guilde_acc:
        torch.save(partial_order, save_location + option_save_location + 'Manual_Model_no_loss_' + str(thres) +'_mask_v2.pth')
        torch.save(net.state_dict(), save_location + option_save_location+'Manual_Model_no_loss_' + str(thres) +'_v2.pth')
        print('save parameter path: ', save_location + option_save_location+'Manual_Model_no_loss_' + str(thres) +'_v2.pth')
        sys.exit("find best combanation")

      # if thres >= 1:
      #   thres = thres / 2
      # else:
      thres = thres - step
      last_order = partial_order
      del net
    
    # # # # Verify result:
    # # thres = 3.314325 for previous version & 3.384175 for potential new case?
    thres = thres
    test = search_range
    partial_order = torch.load(save_location + option_save_location + 'Manual_Model_no_loss_' + str(thres) +'_mask_v2.pth', map_location=torch.device(device)) 
    para = torch.load(save_location + option_save_location+'Manual_Model_no_loss_' + str(thres) +'_v2.pth', map_location=torch.device(device))
    net = res.ResNet50(res.Bottleneck,[3,4,6,3], replace_point=test, apply_ratio=ar, partial=partial_order)
    net.load_state_dict(para, strict=True)   
    net = net.to(device) 
    acc = validation(net, criterion)
    print('verify acc: ',  acc, '%')
