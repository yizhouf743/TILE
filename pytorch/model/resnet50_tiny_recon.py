from dataclasses import replace
from multiprocessing.dummy import Array
import re
from numpy import array
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import numpy as np

__all__ = ['ResNet', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class smooth_unit(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1, base_width=64, dilation=1, norm_layer=None,
        replace_layer = None, repeat_size=2, chan_per_cn=2, apply_ratio=None, partial_order=None):
        super(smooth_unit, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups !=1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")

        self.conv1 = nn.Conv2d(inplanes, planes, 1, stride, bias=False)
        self.bn1 = norm_layer(planes)  
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)    
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.dowansample = downsaple
        self.stride = stride
        
        self.repeat_size = repeat_size
        self.replace_layer = np.array(replace_layer)
        self.cn = chan_per_cn
        
        self.apply_ratio = apply_ratio
        self.order_list = partial_order
        
 
    def forward(self, x: Tensor):
        identity = x
        
        if 0 in self.replace_layer:      
          if self.apply_ratio[0] == 0:
            x = F.avg_pool3d(x, kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
            x = x.repeat_interleave(self.cn, dim=1)
          else:
            if self.order_list[0] is not None:
                mask_x = torch.ones_like(x)
                target_chan = torch.as_tensor(self.order_list[0]).tolist()
                x1 = F.avg_pool3d(x.clone(), kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
                x1 = x1.repeat_interleave(self.cn, dim=1)   
                mask_x[:, target_chan, :, :] = 0
                x = x * mask_x + (1 - mask_x) * x1
                del x1, mask_x, target_chan
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if 1 in self.replace_layer:    
            if self.apply_ratio[1] == 0:
                pad = 0
                out = F.avg_pool2d(out, kernel_size=self.repeat_size, stride=self.repeat_size, padding=pad)
                out = out.repeat_interleave(self.repeat_size, dim=3)
                out = out.repeat_interleave(self.repeat_size, dim=2)    
            else:
                if self.order_list[1] is not None:
                    mask_out = torch.ones_like(out)
                    target_chan = torch.as_tensor(self.order_list[1]).tolist()  
                    out1 = F.avg_pool2d(out.clone(), kernel_size=self.repeat_size, stride=self.repeat_size, padding=0)
                    out1 = out1.repeat_interleave(self.repeat_size, dim=3)
                    out1 = out1.repeat_interleave(self.repeat_size, dim=2)     
                    mask_out[:, target_chan, :, :] = 0
                    out = out * mask_out + (1 - mask_out) * out1
                    del out1, mask_out, target_chan 
          
        out = self.conv2(out) 
        out = self.bn2(out)
        out = self.relu(out)
        
        if 2 in self.replace_layer:
          if self.apply_ratio[1] == 0:
            out = F.avg_pool3d(out, kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
            out = out.repeat_interleave(self.cn, dim=1)
          else:
            if self.order_list[2] is not None:
                mask_out = torch.ones_like(out)
                target_chan = torch.as_tensor(self.order_list[2]).tolist()  
                out1 = F.avg_pool3d(out.clone(), kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
                out1 = out1.repeat_interleave(self.cn, dim=1)    
                mask_out[:, target_chan, :, :] = 0
                # out = out * mask_out + (mask_out - 1).abs() * out1
                out = out * mask_out + (1 - mask_out) * out1
                del out1, mask_out, target_chan

        out = self.conv3(out) 
        out = self.bn3(out)

        if self.dowansample is not None:            
          identity = self.dowansample(x)

        out += identity
        out = self.relu(out)
        return out
    
class smooth_unit2(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1, base_width=64, dilation=1, norm_layer=None,
        replace_layer = None, repeat_size=2, chan_per_cn=2, apply_ratio=None, partial_order=None):
        super(smooth_unit2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups !=1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")

        self.conv1 = nn.Conv2d(inplanes, planes, 1, stride, bias=False)
        self.bn1 = norm_layer(planes)  
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)    
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.dowansample = downsaple
        self.stride = stride
        
        self.repeat_size = repeat_size
        self.replace_layer = np.array(replace_layer)
        self.cn = chan_per_cn
        
        self.apply_ratio = apply_ratio
        self.order_list = partial_order
        
 
    def forward(self, x: Tensor):
        identity = x
        
        if 0 in self.replace_layer:      
          if self.apply_ratio[0] == 0:
            x = F.avg_pool3d(x, kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
            x = x.repeat_interleave(self.cn, dim=1)
          else:
            if self.order_list[0] is not None:
                mask_x = torch.ones_like(x)
                target_chan = torch.as_tensor(self.order_list[0]).tolist()
                x1 = F.avg_pool3d(x.clone(), kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
                x1 = x1.repeat_interleave(self.cn, dim=1)   
                mask_x[:, target_chan, :, :] = 0
                x = x * mask_x + (1 - mask_x) * x1
                del x1, mask_x, target_chan
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if 1 in self.replace_layer:    
            if self.apply_ratio[1] == 0:
                out = F.avg_pool3d(out, kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
                out = out.repeat_interleave(self.cn, dim=1) 
            else:
                if self.order_list[1] is not None:
                    mask_out = torch.ones_like(out)
                    target_chan = torch.as_tensor(self.order_list[2]).tolist()  
                    out1 = F.avg_pool3d(out.clone(), kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
                    out1 = out1.repeat_interleave(self.cn, dim=1)    
                    mask_out[:, target_chan, :, :] = 0
                    out = out * mask_out + (1 - mask_out) * out1
                    del out1, mask_out, target_chan
          
        out = self.conv2(out) 
        out = self.bn2(out)
        out = self.relu(out)
        
        if 2 in self.replace_layer:
          if self.apply_ratio[1] == 0:
            out = F.avg_pool3d(out, kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
            out = out.repeat_interleave(self.cn, dim=1)
          else:
            if self.order_list[2] is not None:           
                mask_out = torch.ones_like(out)
                target_chan = torch.as_tensor(self.order_list[2]).tolist()  
                out1 = F.avg_pool3d(out.clone(), kernel_size=(self.cn, 1, 1), stride=(self.cn, 1, 1))
                out1 = out1.repeat_interleave(self.cn, dim=1)    
                mask_out[:, target_chan, :, :] = 0
                out = out * mask_out + (1 - mask_out) * out1
                del out1, mask_out, target_chan

        out = self.conv3(out) 
        out = self.bn3(out)

        if self.dowansample is not None:            
          identity = self.dowansample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups !=1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")

        self.conv1 = nn.Conv2d(inplanes, planes, 1, stride, bias=False)
        self.bn1 = norm_layer(planes)  
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)    
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.dowansample = downsaple
        self.stride = stride
        
    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) 
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) 
        out = self.bn3(out)

        if self.dowansample is not None:            
            identity = self.dowansample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=200, replace_point=[], cn=[2, 2, 2, 2], apply_ratio=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], partial=None):
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.chan_per_cn = cn
        self.apply_ratio = apply_ratio
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.partial = partial
        
        if len(replace_point) > 0:
            replace_point = np.squeeze(replace_point)
            if np.array(replace_point).size > 1:
                max_point = max(replace_point)
            else:
                max_point = replace_point
                replace_point =[replace_point, 128, 256, 512, 1024]
                
            if max_point < 9:
                replace_range = [x for x in replace_point if x in list(range(9))]
                self.layer1 = self._block_replace(block, 64, layers[0], replace_point=replace_range, cn=self.chan_per_cn[0], ar=self.apply_ratio[0], partial=self.partial[:9])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
            elif max_point < 21:
                replace_range = [x for x in replace_point if x in list(range(9))]
                self.layer1 = self._block_replace(block, 64, layers[0], replace_point=replace_range, cn=self.chan_per_cn[0], ar=self.apply_ratio[0], partial=self.partial[:9])
                replace_range = [x - 9 for x in replace_point if x in list(range(9, 21))]
                self.layer2 = self._block_replac(block, 128, layers[1], stride=2, replace_point=replace_range, cn=self.chan_per_cn[1], ar=self.apply_ratio[1], partial=self.partial[9:21])
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
            elif max_point < 39:
                replace_range = [x for x in replace_point if x in list(range(9))]
                self.layer1 = self._block_replace(block, 64, layers[0], replace_point=replace_range, cn=self.chan_per_cn[0], ar=self.apply_ratio[0], partial=self.partial[:9])
                replace_range = [x - 9 for x in replace_point if x in list(range(9, 21))]
                self.layer2 = self._block_replace(block, 128, layers[1], stride=2, replace_point=replace_range, cn=self.chan_per_cn[1], ar=self.apply_ratio[1], partial=self.partial[9:21])
                replace_range = [x - 21 for x in replace_point if x in list(range(21, 39))]
                self.layer3 = self._block_replace(block, 256, layers[2], stride=2, replace_point=replace_range, cn=self.chan_per_cn[2], ar=self.apply_ratio[2], partial=self.partial[21:39])
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
            else:
                replace_range = [x for x in replace_point if x in list(range(9))]
                self.layer1 = self._block_replace(block, 64, layers[0], replace_point=replace_range, cn=self.chan_per_cn[0], ar=self.apply_ratio[0], partial=self.partial[:9])
                replace_range = [x - 9 for x in replace_point if x in list(range(9, 21))]
                self.layer2 = self._block_replace(block, 128, layers[1], stride=2, replace_point=replace_range, cn=self.chan_per_cn[1], ar=self.apply_ratio[1], partial=self.partial[9:21])
                replace_range = [x - 21 for x in replace_point if x in list(range(21, 39))]
                self.layer3 = self._block_replace(block, 256, layers[2], stride=2, replace_point=replace_range, cn=self.chan_per_cn[2], ar=self.apply_ratio[2], partial=self.partial[21:39])
                replace_range = [x - 39 for x in replace_point if x in list(range(39, 48))]
                self.layer4 = self._block_replace(block, 512, layers[3], stride=2, replace_point=replace_range, cn=self.chan_per_cn[3], ar=self.apply_ratio[3], partial=self.partial[39:], sp_case=True)    
        else:
          self.layer1 = self._make_layer(block, 64, layers[0])
          self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
          self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
          self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
          
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(6*6*512 * block.expansion, num_classes)
        
    
    def _block_replace(self, block, planes, blocks, stride=1, replace_point=[], repeat_size=2, cn=2, ar=None, partial=None, sp_case=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        
        print(replace_point, blocks, ar)
        if any(x in replace_point for x in list(range(3))):   
          replace_layer = [x for x in replace_point if x in list(range(3))]
          print(replace_layer)
          layers.append(smooth_unit(self.inplanes, planes, stride, downsample, replace_layer=replace_layer, repeat_size=repeat_size, chan_per_cn=cn, apply_ratio=ar[:2], partial_order=partial[:3]))
        else: 
          layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            select_range = list(range(3 * _, 3 * (_ + 1)))
            if replace_point:
                if any(x in replace_point for x in select_range):
                    replace_layer = [x for x in replace_point if x in select_range]
                    replace_layer = (np.squeeze(replace_layer) % 3).tolist()
                    print(replace_layer)
                    
                    if _ < blocks - 1:
                        pr = partial[3 * _: 3 * (_ + 1) + 1]
                    else:
                        pr = partial[3 * _: 3 * (_ + 1)]

                    if sp_case is False:
                        m = smooth_unit(self.inplanes, planes, replace_layer=replace_layer, repeat_size=repeat_size, chan_per_cn=cn, apply_ratio=ar[2:], partial_order=pr)
                    else:
                        m = smooth_unit2(self.inplanes, planes, replace_layer=replace_layer, repeat_size=repeat_size, chan_per_cn=cn, apply_ratio=ar[2:], partial_order=pr)
                else:
                    m = block(self.inplanes, planes)
            else:
                m = block(self.inplanes, planes)
            layers.append(m)
        return nn.Sequential(*layers) 
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x
