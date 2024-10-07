import torch, copy
import torch.nn as nn

def cal_importance(net):
   pre_trained_model = net.state_dict()
   key = list(pre_trained_model.keys())
   im_score = []
   
   for i in range(len(key)):
    if 'conv.weight' in key[i]:
      layer_weight = pre_trained_model[key[i]]
      stat = layer_weight.sum(dim=(2,3))
      fs = stat.size(0) * stat.size(1)
      spar = torch.count_nonzero(stat) / fs
      # print("non-prun rate after retrain", key[i], spar)     
      score = torch.abs(layer_weight).sum()/(layer_weight.shape[0] * layer_weight.shape[1])
      score = score * spar
      im_score.append(score)
   im_score = torch.stack(im_score)
   im_score = torch.nn.functional.normalize(im_score, p=2.0, dim=0)
   print("important score for each layer: ", im_score)
   return im_score

## For Chann Prun method:
def cal_bn_importance(net):
   pre_trained_model = net.state_dict()
   key = list(pre_trained_model.keys())
   im_score = []
   bn_score = []
   
   for i in range(len(key)):
    if 'bn.weight' in key[i]:
      layer_weight = pre_trained_model[key[i]]
      spar = torch.count_nonzero(layer_weight) / layer_weight.shape[0]    
      score = torch.abs(layer_weight).sum() / layer_weight.shape[0]
      score = score * spar
      im_score.append(score)

   im_score = torch.stack(im_score)
   im_score = torch.nn.functional.normalize(im_score, p=2.0, dim=0)
   print("important score for each layer: ", im_score)
   return im_score


def cal_conv_importance(net):
   pre_trained_model = net.state_dict()
   key = list(pre_trained_model.keys())
   im_score = []
   
   for i in range(len(key)):
    if 'conv.weight' in key[i]:
      layer_weight = pre_trained_model[key[i]]
      stat = layer_weight
      fs = stat.size(0) * stat.size(1) * stat.size(2) * stat.size(3)
      spar = torch.count_nonzero(stat) / fs
      # print("non-prun rate after retrain", key[i], spar)     
      score = torch.abs(layer_weight).sum()/(layer_weight.shape[0] * layer_weight.shape[1] * layer_weight.shape[2])
      score = score * spar
      im_score.append(score)
   im_score = torch.stack(im_score)
   im_score = torch.nn.functional.normalize(im_score, p=2.0, dim=0)
   print("important score for each layer: ", im_score)
   return im_score
   
def cal_grad_importance(net):
   pre_trained_model = net.state_dict()
   grad_dict = {k:v.grad for k,v in net.named_parameters()}
   
   key = list(pre_trained_model.keys())
   im_score = []

   for i in range(len(key)):
    if 'conv.weight' in key[i]:
      layer_weight = grad_dict[key[i]]
      stat = layer_weight
      score = torch.abs(layer_weight).sum()
      im_score.append(score)
   im_score = torch.stack(im_score)
   im_score = torch.nn.functional.normalize(im_score, p=2.0, dim=0)
   return grad_dict, im_score

def cal_subgrad(flatten_x, flatten_grad_f, lmbda):
  flatten_subgrad_reg = torch.zeros_like(flatten_grad_f)
  norm = torch.norm(flatten_x, p=2, dim=1)
  non_zero_mask = norm != 0
  flatten_subgrad_reg[non_zero_mask] = flatten_x[non_zero_mask] / (norm[non_zero_mask] + 1e-7).unsqueeze(1)
  flatten_cal_subgrad = flatten_grad_f + lmbda * flatten_subgrad_reg
  return flatten_cal_subgrad
    
def get_momentum_grad(param_state, key, momentum, grad):
    if momentum > 0:
        if key not in param_state:
            buf = param_state[key] = grad
        else:
            buf = param_state[key]
            buf.mul_(momentum).add_(grad)
        return buf
    else:
        return grad 

def grad_descent_update(x, lr, grad):
    return x - lr * grad
    

def half_space_project(hat_x, x, x_base, epsilon, name, previous_group_sparsity, threshold, show_result=False):
    num_groups = x.numel()
    ## half space project:
    ## x == x_base is True!!    
    x_norm = torch.norm(x_base, p=2, dim=1)
    decision_signal = 0
    ## w_1 * w_b < epilson * ||w_b||^2:        
    proj_idx = torch.abs(torch.bmm(hat_x.view(hat_x.shape[0], 1, -1), x_base.view(x_base.shape[0], -1, 1)).squeeze() / x_norm ** 2)
    proj_idx = torch.nan_to_num(proj_idx)
    trial_group_sparsity = torch.sum(proj_idx) / float(num_groups)
    trial_group_sparsity = torch.sqrt(trial_group_sparsity)
    trial_group_sparsity = torch.nan_to_num(trial_group_sparsity)
    
    if show_result == True:
      print('inital_result: ', name, trial_group_sparsity)
    
    if trial_group_sparsity <= threshold:
       decision_signal = 1
    
    name_list = list(previous_group_sparsity.keys())
    if name not in name_list:
      previous_group_sparsity[name] = trial_group_sparsity
    else:
      previous_group_sparsity[name] += trial_group_sparsity
    return decision_signal
    
def partial_order(hat_x, x, x_base, epsilon, name, partial):
    x_norm = torch.norm(x_base, p=2, dim=0)
    proj_idx = torch.abs(torch.bmm(hat_x.view(hat_x.shape[1], 1, -1), x_base.view(x_base.shape[1], -1, 1)).squeeze() / x_norm ** 2)
    # print(name, proj_idx.shape, x.shape, x_base.shape, x_norm.shape)
    proj_idx = torch.nan_to_num(proj_idx).squeeze()
    proj_idx = proj_idx.view(-1, 9).sum(dim=1) / (9 * x_base.shape[0]) # 9 is the filter size (i,e. f ^ 2)
    proj_idx = torch.sqrt(proj_idx)
    proj_idx = torch.nan_to_num(proj_idx).squeeze()
    print(proj_idx.shape)
    name_list = list(partial.keys())

    if name not in name_list:
      partial[name] = proj_idx    
    else:
      old = partial[name]
      partial[name] = torch.add(old, proj_idx)


def hspg(net, baseline_net, grad_dict, lmbda, epsilon, optimizer, threshold, previous_group_sparsity=None, partial=None, show_result=False):
    lr = optimizer.param_groups[0]['lr']
    momentum = optimizer.param_groups[0]['momentum']
    
    # replace_list = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14]
    layer_key = list(net.state_dict().keys())
    param_dict = {k:v.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
    base_dict = {k:v.data.view(v.data.shape[0], -1) for k,v in baseline_net.named_parameters()}
    
    # new_list = []
    for name in layer_key:
      if 'conv.weight' in name:
        if 'features.0.conv.weight' not in name:
          flatten_x = param_dict[name]
          grad = grad_dict[name]
          flatten_grad_f = grad
          x = base_dict[name]
          ## gradient descent update:
          flatten_cal_subgrad = cal_subgrad(flatten_x, flatten_grad_f, lmbda)
          flatten_cal_subgrad = get_momentum_grad(grad_dict, name, momentum, flatten_cal_subgrad)      
          # compute trial iterate
          flatten_hat_x = grad_descent_update(flatten_x, lr, flatten_cal_subgrad)
          # decision_signal = half_space_project(flatten_hat_x, flatten_x, x, epsilon, name, previous_group_sparsity, threshold, show_result)
          partial_order(flatten_hat_x, flatten_x, x, epsilon, name, partial)
          # new_list.extend([decision_signal])
    # new_test = [i for i,j in zip(replace_list,new_list) if j == 1]
    # return new_test    
    


def hspg_resnet34(net, baseline_net, grad_dict, lmbda, epsilon, optimizer, threshold, previous_group_sparsity=None, partial=None, show_result=False):
    lr = optimizer.param_groups[0]['lr']
    momentum = optimizer.param_groups[0]['momentum']
    replace_list = list(range(32))
    layer_key = list(net.state_dict().keys())
    param_dict = {k:v.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
    base_dict = {k:v.data.view(v.data.shape[0], -1) for k,v in baseline_net.named_parameters()}
    
    new_list = []
    key_word = ['layer', 'conv', 'weight']
    for name in layer_key:
      if all(c in name for c in key_word):
        flatten_x = param_dict[name]
        grad = grad_dict[name]
        flatten_grad_f = grad
        x = base_dict[name]
        ## gradient descent update:
        flatten_cal_subgrad = cal_subgrad(flatten_x, flatten_grad_f, lmbda)
        flatten_cal_subgrad = get_momentum_grad(grad_dict, name, momentum, flatten_cal_subgrad)      
        # compute trial iterate
        flatten_hat_x = grad_descent_update(flatten_x, lr, flatten_cal_subgrad)
        decision_signal = half_space_project(flatten_hat_x, flatten_x, x, epsilon, name, previous_group_sparsity, threshold, show_result)
        partial_order(flatten_hat_x, flatten_x, x, epsilon, name, partial)
        new_list.extend([decision_signal])
    new_test = [i for i,j in zip(replace_list,new_list) if j == 1]
    return new_test 
