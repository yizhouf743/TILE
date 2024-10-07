import torch, copy
import torch.nn as nn

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

    

def partial_order(hat_x, x, x_base, epsilon, name, partial):
    if 'conv2' in name:
      x_norm = torch.norm(x_base, p=2, dim=0)
      proj_idx = torch.abs(torch.bmm(hat_x.view(hat_x.shape[1], 1, -1), x_base.view(x_base.shape[1], -1, 1)).squeeze() / x_norm ** 2)
      # proj_idx = proj_idx.view(x_base.shape[0], -1).sum(dim=1) / 9 # 9 is the filter size (i,e. f ^ 2)
      proj_idx = proj_idx.view(-1, 9).sum(dim=1) / 9 # 9 is the filter size (i,e. f ^ 2)
      proj_idx = torch.nan_to_num(proj_idx).squeeze()
    else:       
      x_norm = torch.norm(x_base, p=2, dim=0)
      proj_idx = torch.abs(torch.bmm(hat_x.view(hat_x.shape[1], 1, -1), x_base.view(x_base.shape[1], -1, 1)).squeeze() / x_norm ** 2)
      proj_idx = torch.nan_to_num(proj_idx).squeeze()
  
    name_list = list(partial.keys())
    
    if name not in name_list:
      partial[name] = proj_idx    
    else:
      old = partial[name]
      partial[name] = torch.add(old, proj_idx)

def hspg_resnet50(net, baseline_net, grad_dict, lmbda, epsilon, optimizer, threshold, previous_group_sparsity=None, partial=None, show_result=False):
    lr = optimizer.param_groups[0]['lr']
    momentum = optimizer.param_groups[0]['momentum']
    replace_list = list(range(48))
    layer_key = list(net.state_dict().keys())
    param_dict = {k:v.data.view(v.data.shape[0], -1) for k,v in net.named_parameters()}
    base_dict = {k:v.data.view(v.data.shape[0], -1) for k,v in baseline_net.named_parameters()}
    
    
    new_list = []
    for name in layer_key:
      if 'dowansample' not in name:
        if all([x in name for x in ['layer', 'conv', 'weight']]):
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
          if any([x in name for x in ['conv1', 'conv2', 'conv3']]):
            partial_order(flatten_hat_x, flatten_x, x, epsilon, name, partial)
          new_list.extend([decision_signal])
#    new_list = [i % 3 for i in new_list]
    new_test = [i for i,j in zip(replace_list,new_list) if j == 1]
    return new_test  
            
