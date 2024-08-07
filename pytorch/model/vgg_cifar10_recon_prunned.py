from typing import Union, List, Dict, Any, cast
import torch, math
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

default_pool = [0, 2, 2, 2, 'M', 2, 2, 2, 2, 2, 2, 'M', 2, 2, 2,'M']
default_tile = ['N', 0, 0, 0, 'N', 0, 0, 0, 0, 0, 0, 'N', 1, 1, 1,'N']


class mixed_unit(nn.Module):
    def __init__(self, conv2d, v, repeat_size, partial_order, next_type) -> None:
        super(mixed_unit, self).__init__()
        self.repeat_size = repeat_size
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv2d
        self.inital_step = 0
        self.bn = nn.BatchNorm2d(v)
        self.order_list = partial_order
        self.next = next_type

    def forward(self, x):
        if self.order_list[0] is not None:
            mask_out = torch.ones_like(x)
            target_chan = torch.as_tensor(self.order_list[0]).tolist()  
            out1 = F.avg_pool2d(x.clone(), kernel_size=self.repeat_size, stride=self.repeat_size, padding=0)
            out1 = out1.repeat_interleave(self.repeat_size, dim=3)
            out1 = out1.repeat_interleave(self.repeat_size, dim=2)     
            mask_out[:, target_chan, :, :] = 0
            x = x * mask_out + (1 - mask_out) * out1
        out = self.bn(self.conv(x))

        if len(self.order_list) > 1:
            if self.next == 0:
                mask_out = torch.ones_like(out)
                target_chan = torch.as_tensor(self.order_list[1]).tolist()  
                out1 = F.avg_pool2d(out.clone(), kernel_size=self.repeat_size, stride=self.repeat_size, padding=0)
                out1 = out1.repeat_interleave(self.repeat_size, dim=3)
                out1 = out1.repeat_interleave(self.repeat_size, dim=2)     
                mask_out[:, target_chan, :, :] = 0
                out = out * mask_out + (1 - mask_out) * out1
                del out1, mask_out, target_chan 
            elif self.next == 1:
                mask_out = torch.ones_like(out)
                target_chan = torch.as_tensor(self.order_list[1]).tolist()  
                out1 = F.avg_pool3d(out.clone(), kernel_size=(self.repeat_size, 1, 1), stride=(self.repeat_size, 1, 1))
                out1 = out1.repeat_interleave(self.repeat_size, dim=1)    
                mask_out[:, target_chan, :, :] = 0
                out = out * mask_out + (1 - mask_out) * out1
                del out1, mask_out, target_chan
        out = self.relu(out)
        return out

class mixed_unit2(nn.Module):
    def __init__(self, conv2d, v, repeat_size, partial_order, next_type) -> None:
        super(mixed_unit2, self).__init__()
        self.repeat_size = repeat_size
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv2d
        self.inital_step = 0
        self.bn = nn.BatchNorm2d(v)
        self.order_list = partial_order
        self.next = next_type

    def forward(self, x):
        if self.order_list[0] is not None:
            mask_out = torch.ones_like(x)
            target_chan = torch.as_tensor(self.order_list[0]).tolist()  
            out1 = F.avg_pool3d(x.clone(), kernel_size=(self.repeat_size, 1, 1), stride=(self.repeat_size, 1, 1))
            out1 = out1.repeat_interleave(self.repeat_size, dim=1)    
            mask_out[:, target_chan, :, :] = 0
            x = x * mask_out + (1 - mask_out) * out1
            del out1, mask_out, target_chan
        out = self.bn(self.conv(x))

        if len(self.order_list) > 1:
            if self.next == 0:
                mask_out = torch.ones_like(out)
                target_chan = torch.as_tensor(self.order_list[1]).tolist()  
                out1 = F.avg_pool2d(out.clone(), kernel_size=self.repeat_size, stride=self.repeat_size, padding=0)
                out1 = out1.repeat_interleave(self.repeat_size, dim=3)
                out1 = out1.repeat_interleave(self.repeat_size, dim=2)     
                mask_out[:, target_chan, :, :] = 0
                out = out * mask_out + (1 - mask_out) * out1
                del out1, mask_out, target_chan 
            elif self.next == 1:
                mask_out = torch.ones_like(out)
                target_chan = torch.as_tensor(self.order_list[1]).tolist()  
                out1 = F.avg_pool3d(out.clone(), kernel_size=(self.repeat_size, 1, 1), stride=(self.repeat_size, 1, 1))
                out1 = out1.repeat_interleave(self.repeat_size, dim=1)    
                mask_out[:, target_chan, :, :] = 0
                out = out * mask_out + (1 - mask_out) * out1
                del out1, mask_out, target_chan
        out = self.relu(out)
        return out

class normal_unit(nn.Module):
    def __init__(self, conv2d, v) -> None:
        super(normal_unit, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv2d
        self.bn = nn.BatchNorm2d(v)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
	        nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
        
def make_layers(cfg, select_range=None, ps=None, tile_type=None, batch_norm=False, partial=[]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    pool_size = ps
    counter = -1 # we not include partial order of conv0, thus inital counter should be -1.
     
    for i in range(len(cfg)):
        v = cfg[i]
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if (select_range) and (i in select_range) and (pool_size[i] != 0):
                if counter + 2 >= len(cfg):
                    select_order = partial[counter]
                    next = None
                else:
                    select_order = partial[counter:counter+1]
                    next = tile_type[i+1]
                    if next  == "N":
                        if i + 2 < len(cfg):
                            next = tile_type[i+2]
                        else:
                            next = None
                print(i, cfg[i], tile_type[i], next)
                if tile_type[i] != "N":
                    if tile_type[i] == 0:
                        layers += [mixed_unit(conv2d, v, pool_size[i], select_order, next)]
                    else:
                        layers += [mixed_unit2(conv2d, v, pool_size[i], select_order, next)]
            else:
                layers += [normal_unit(conv2d, v)]      
            in_channels = v
            counter += 1
    layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "DDP": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "D": [64, 64, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, replace_point:List[int], pool_size:List[int], tile_type:List[int], partial_order, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, select_range=replace_point, ps=pool_size, partial=partial_order), **kwargs)
    return model


def vgg16(pretrained: bool = False, progress: bool = True, replace_point:List[int]=[], pool_size:List[int]=default_pool, tile_type:List[int]=default_tile, partial_order=None, **kwargs: Any) -> VGG:
    return _vgg("vgg16", "D", False, pretrained, progress, replace_point, pool_size, partial_order, **kwargs)
