
  
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
from torch import nn
import logging

from common.pytorch_utils import load_model
from common.model import Model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResEncoder(nn.Module):
    def __init__(self, block=BasicBlock, n_channels=3, load_pretrained=None):
        super(ResEncoder, self).__init__()
        self.n_channels = n_channels
        layers = [2, 2, 2, 2]
        self.inplanes = 64
        self.conv0 = nn.Conv2d(self.n_channels, 8, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(8)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if load_pretrained is not None:
            pass

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x0 = self.relu0(x)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        # x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def copy_params_from_pretrained_model(self, pretrained_model):
        logging.info("Copying weights from pretrained model...")
        names = ['conv1'] + ['layer{}'.format(i) for i in range(1, 4)]
        for name in names:
            l1 = getattr(self, name)
            l2 = getattr(pretrained_model, name)
            for m1, m2 in zip(l1.modules(), l2.modules()):
                if isinstance(m1, nn.Conv2d):
                    assert(isinstance(m2, nn.Conv2d))
                    assert m1.weight.size() == m2.weight.size()
                    m1.weight.data = m2.weight.data
                    if m1.bias is not None and m2.bias is not None:
                        assert m1.bias.size() == m2.bias.size()
                        m1.bias.data = m2.bias.data

    def freeze_params(self):
        logging.info("Freeze encoder(ResEncoder) weights...")
        names = ['conv1'] + ['layer{}'.format(i) for i in range(1, 4)]
        for name in names:
            l1 = getattr(self, name)
            for m1 in l1.modules():
                if isinstance(m1, nn.Conv2d):
                    m1.weight.required_grad = False
                    if m1.bias is not None:
                        m1.bias.required_grad = False

    def print_features(self):
        child_counter = 0
        names = ['conv1'] + ['layer{}'.format(i) for i in range(1, 5)]
        for name in names:
            child = getattr(self, name)
            logging.info(name)
            for m in child.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d):
                    logging.info(m)
            child_counter += 1


class ResNetSide(Model):
    name = "ResNetSide"
    def __init__(self, batch_size, config, use_gpu=False):
        super(ResNetSide, self).__init__()
        block = BasicBlock
        num_classes = 2
        zero_init_residual = False
        self.batch_size = batch_size
        self.config = config
        self.use_gpu = use_gpu
        self.n_channels = self.config["N_CHANNELS"]
        self.encoder = ResEncoder(block, self.n_channels)
        self.layer4 = self.encoder._make_layer(block, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config["LEARNING_RATE"],
            momentum=self.config["MOMENTUM"],
            weight_decay=self.config["WEIGHT_DECAY"]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        if self.config["PRETRAINED_LOAD"] is not None:
            self._load_shared_encoder(self.config["PRETRAINED_LOAD"])
            self.encoder.freeze_params()

    

    def forward(self, x):
        [_, _, _, _, x] = self.encoder.forward(x)
        # logging.info("{}, {}, {}, {}".format(x1.size(), x2.size(), x3.size(), x.size()))
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        return x
    
    def input_desc(self):
        return [
            torch.from_numpy(np.random.rand(self.batch_size, self.n_channels, 96, 96)).float(),
            torch.from_numpy(np.random.randint(low=0, high=1, size=(self.batch_size)).astype(np.int32)).long(),
        ]

    def run_step(self, data):
        data = self.process_input_data(data)
        img = data[0]
        side_gt = data[1]
        self.optimizer.zero_grad()
        side_pred = self.forward(img)
        loss = self.criterion(side_pred, side_gt)
        self.params["loss"] = loss
        loss.backward()
        self.optimizer.step()

    def process_input_data(self, data):
        img = self.process_img(data[0])
        # img = data[0].type(torch.FloatTensor)
        side_class = data[1].type(torch.LongTensor)
        if self.use_gpu:
            # img = img.cuda()
            side_class = side_class.cuda()
        return [img, side_class]
    
    def process_img(self, img):
        img = img.type(torch.FloatTensor)
        if self.use_gpu:
            img = img.cuda()
        return img