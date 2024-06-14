import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
#class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#
#    def forward(self, x):
#        ...
#
######################################################
class ASHBaseResNet18(nn.Module):
    # activation_layer is which layer used/ layer_type is what layer used(Conv2d;BatchNorm2d;ReLU;MaxPool2d;)
    def __init__(self, activation_layer=1, layer_type=nn.Conv2d):
        super(ASHBaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.save_hooks = []
        # self.activation_maps = {}
        self._register_activation_shaping_hooks(activation_layer, layer_type)

    def _register_activation_shaping_hooks(self, activation_layer, layer_type):

        count = 0
        for name, module in self.resnet.named_modules():
            if isinstance(module, layer_type):
                count += 1
                # if count % activation_layer == 0:
                if count == activation_layer:
                #if count % activation_layer ==0 and count == 6 or count == 8:
                    hook = module.register_forward_hook(self.activation_shaping_hook)
                    self.save_hooks.append(hook)
                    print(f"Activation hook added:,{name}")

    def activation_shaping_hook(self, module, input, output):
        M = torch.where(torch.rand_like(output) < 1.0, 0.0, 1.0)
        A_binarized = (output > 0).float()
        M_binarized = (M > 0).float()
        new_output = A_binarized * M_binarized
        return new_output
    def forward(self, x):
        return self.resnet(x)
class ASHDAResNet18(nn.Module):  # task 3
    # activation_layer is which layer used/ layer_type is what layer used(Conv2d;BatchNorm2d;ReLU;MaxPool2d;)
    def __init__(self, activation_layer=15 , layer_type=nn.Conv2d):
        super(ASHDAResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.save_hooks = []
        self.activation_maps = {}
        self._register_activation_shaping_hooks(activation_layer, layer_type)

    def _register_activation_shaping_hooks(self, activation_layer, layer_type):

        count = 0
        for name, module in self.resnet.named_modules():
            if isinstance(module, layer_type):
                count += 1
                # if count % activation_layer == 0:
                if count == activation_layer:
                # if count % activation_layer ==0 and count == 6 or count == 8:
                    self.create_hook(name, module)
                    # print(f"Activation hook added:,{name}")
    def create_hook(self, name, module):
        print("Registering hook on:", name)
        hook = module.register_forward_hook(lambda module, input, output: self.save_activation_map(name, module, input, output))
        self.save_hooks.append(hook)
    def save_activation_map(self, name, module, input, output):
        new_output = self.activation_shaping_hook(module, input, output)
        self.activation_maps[name] = new_output
        #print(f"Storing activation map for layer: {name}")
    def activation_shaping_hook(self, module, input, output):
           M = torch.where(torch.rand_like(output) < 0.5, 0.0, 1.0)
           # print("M:", M)
           A_binarized = (output > 0).float()
           M_binarized = (M > 0).float()
           # print("A_binarized:", A_binarized)
           # print("output:", output)
           # print("M_binarized:", M_binarized)
           new_output = A_binarized * M_binarized
           return new_output

    def forward(self, x, activation_map=None):
        #print(f"activation_map5: {activation_map}")
        def apply_activation_maps(module_name, x):
            if activation_map and module_name in activation_map:
                #
                # print(f"Applying activation map on {module_name}, shape: {activation_map[module_name].shape}")
                #print(f"activation_map1: {activation_map[module_name]}")
                x = x * activation_map[module_name]
            return x

        x = self.resnet.conv1(x).clone()
        x = apply_activation_maps('conv1', x)
        x = self.resnet.bn1(x)
        x = apply_activation_maps('bn1', x)
        x = self.resnet.relu(x)
        x = apply_activation_maps('relu', x)
        x = self.resnet.maxpool(x)
        x = apply_activation_maps('maxpool', x)

        def process_layer(layer, layer_name, x):
            for block_idx, block in enumerate(layer):
                block_name = f"{layer_name}.{block_idx}"
                identity = x

                out = block.conv1(x)
                out = apply_activation_maps(f"{block_name}.conv1", out)
                out = block.bn1(out)
                out = apply_activation_maps(f"{block_name}.bn1", out)
                out = block.relu(out)
                out = apply_activation_maps(f"{block_name}.relu", out)
                out = block.conv2(out)
                out = apply_activation_maps(f"{block_name}.conv2", out)
                out = block.bn2(out)
                out = apply_activation_maps(f"{block_name}.bn2", out)

                if block.downsample is not None:
                    identity = x
                    for i, downsample_module in enumerate(block.downsample):
                        identity = downsample_module(identity)
                        identity = apply_activation_maps(f"{block_name}.downsample.{i}", identity)

                x = out + identity
                x = block.relu(x)
            return x

        x = process_layer(self.resnet.layer1, 'layer1', x)

        x = process_layer(self.resnet.layer2, 'layer2', x)

        x = process_layer(self.resnet.layer3, 'layer3', x)
        x = process_layer(self.resnet.layer4, 'layer4', x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x
    def get_activation_maps(self, trg_x):
        self.activation_maps.clear()
        with torch.no_grad():
            self(trg_x)
        return self.activation_maps
