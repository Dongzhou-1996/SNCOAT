import time
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Models.attention_module import AddictiveAttention, DotProdAttention, MultiHeadAttention
from Models.function_modules import SELayer


class ConvNet(nn.Module):
    def __init__(self, input_channels, with_SE=False, vis=False):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-6, momentum=0.05)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-6, momentum=0.05)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-6, momentum=0.05)
        self.conv4 = nn.Conv2d(128, 128, 1, 1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-6, momentum=0.05)
        self.max_pool = nn.MaxPool2d(2)
        if with_SE:
            self.se1 = SELayer(32, 8)
            self.se2 = SELayer(64, 8)
            self.se3 = SELayer(128, 8)

        self.with_SE = with_SE
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, x):
        self.attention_visualization.update({'backbone_input': x})

        if self.with_SE:
            x = F.relu(self.max_pool(self.bn1(self.se1(self.conv1(x)))))
            self.attention_visualization.update({'backbone_layer_1': x})
            x = F.relu(self.max_pool(self.bn2(self.se2(self.conv2(x)))))
            self.attention_visualization.update({'backbone_layer_2': x})
            x = F.relu(self.max_pool(self.bn3(self.se3(self.conv3(x)))))
            self.attention_visualization.update({'backbone_layer_3': x})
            x = F.relu(self.max_pool(self.bn4(self.conv4(x))))
            self.attention_visualization.update({'backbone_layer_4': x})
        else:
            x = F.relu(self.max_pool(self.bn1(self.conv1(x))))
            self.attention_visualization.update({'backbone_layer_1': x})
            x = F.relu(self.max_pool(self.bn2(self.conv2(x))))
            self.attention_visualization.update({'backbone_layer_2': x})
            x = F.relu(self.max_pool(self.bn3(self.conv3(x))))
            self.attention_visualization.update({'backbone_layer_3': x})
            x = F.relu(self.max_pool(self.bn4(self.conv4(x))))
            self.attention_visualization.update({'backbone_layer_4': x})

        if not self.vis:
            self.attention_visualization.clear()
        return x

    def visualization_clear(self):
        self.attention_visualization.clear()

    def visualization(self, vis_dir='', vis_time=None, fmt='pdf'):
        if vis_time is None:
            vis_time = time.time()
        # visualize network
        if len(self.attention_visualization) > 0:
            for k, tensor in self.attention_visualization.items():
                print('=> saving {} attention map'.format(k))
                sample = tensor[0]
                # sample: CxHxW
                sample = sample.permute(1, 2, 0)  # HxWxC
                file_name = os.path.join(vis_dir, '{}_{}.{}'.format(vis_time, k, fmt))
                if k == 'backbone_input':
                    plt.imshow(sample[..., :3].detach().cpu().numpy())
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
                else:
                    attention_map = sample.pow(2).mean(2).detach().cpu().numpy()  # HxW
                    attention_map = cv2.resize(src=attention_map, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
                    plt.imshow(attention_map.astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')



class ResNet(nn.Module):
    def __init__(self, input_channels, block=None, num_blocks=None):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(input_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        return x

    def _make_layer(self, block=None, planes=None, num_blocks=None, stride=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        output = F.relu(output)
        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output += self.shortcut(x)
        output = F.relu(output)
        return output