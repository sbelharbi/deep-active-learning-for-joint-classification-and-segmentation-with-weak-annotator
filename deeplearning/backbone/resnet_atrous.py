"""
Main refs:
https://github.com/YudeWang/deeplabv3plus-pytorch
https://github.com/VainF/DeepLabV3Plus-Pytorch
"""
import sys
import threading
import os
import datetime as dt

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


sys.path.append("..")
sys.path.append("../..")

from shared import check_if_allow_multgpu_mode, announce_msg

# lock for threads to protect the instruction that cause randomness and make
# them
thread_lock = threading.Lock()
# thread-safe.

import reproducibility
import constants

ACTIVATE_SYNC_BN = False
# Override ACTIVATE_SYNC_BN using variable environment in Bash:
# $ export ACTIVATE_SYNC_BN="True"   ----> Activate
# $ export ACTIVATE_SYNC_BN="False"   ----> Deactivate

if "ACTIVATE_SYNC_BN" in os.environ.keys():
    ACTIVATE_SYNC_BN = (os.environ['ACTIVATE_SYNC_BN'] == "True")

announce_msg("ACTIVATE_SYNC_BN was set to {}".format(ACTIVATE_SYNC_BN))

if check_if_allow_multgpu_mode() and ACTIVATE_SYNC_BN:  # Activate Synch-BN.
    from deeplearning.syncbn import nn as NN_Sync_BN
    BatchNorm2d = NN_Sync_BN.BatchNorm2d
    announce_msg("Synchronized BN has been activated. \n"
                 "MultiGPU mode has been activated. "
                 "{} GPUs".format(torch.cuda.device_count()))
else:
    BatchNorm2d = nn.BatchNorm2d
    if check_if_allow_multgpu_mode():
        announce_msg("Synchronized BN has been deactivated.\n"
                     "MultiGPU mode has been activated. "
                     "{} GPUs".format(torch.cuda.device_count()))
    else:
        announce_msg("Synchronized BN has been deactivated.\n"
                     "MultiGPU mode has been deactivated. "
                     "{} GPUs".format(torch.cuda.device_count()))

ALIGN_CORNERS = True


__all__ = ['resnet50_atrous', 'resnet101_atrous', 'resnet152_atrous']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

bn_mom = 0.0003


def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1*atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1*atrous, dilation=atrous, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Atrous(nn.Module):

    def __init__(self, arch, block, layers, atrous=None, os=16):
        """
        Atrous resnet.
        :param block:class of the block.
        :param layers: list.
        :param atrous: list or None.
        :param os: int. output stride.
        """
        super(ResNet_Atrous, self).__init__()

        self.name = arch

        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError(
                'resnet_atrous.py: output stride=%d is not supported.' % os)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = BatchNorm2d(64, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1],
                                       stride=stride_list[0])
        self.layer3 = self._make_layer(block, 512, 256, layers[2],
                                       stride=stride_list[1], atrous=16 // os)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3],
                                       stride=stride_list[2],
                                       atrous=[item * 16 // os for item in
                                               atrous])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1,
                    atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0],
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, stride=1,
                                atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return low_level_features, x

    def get_nbr_params(self):
        """
        Compute the number of parameters of the model.
        :return:
        """
        return sum([p.numel() for p in self.parameters()])

    def __str__(self):
        return "{}: RESNET ATROUS.".format(self.name)


def _resnet(arch, block, layers, atrous, os, pretrained, progress, **kwargs):
    model = ResNet_Atrous(
        arch=arch, block=block, layers=layers, atrous=atrous, os=os, **kwargs)
    if pretrained:
        old_dict = load_state_dict_from_url(
            url=model_urls[arch], model_dir='./pretrained-imgnet',
            map_location=torch.device('cpu'), progress=progress)
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_atrous(pretrained=True, os=16, **kwargs):
    """Constructs an atrous ResNet-50 model."""
    return _resnet(arch="resnet50", block=Bottleneck, layers=[3, 4, 6, 3],
                   atrous=[1, 2, 1], os=os, pretrained=pretrained,
                   progress=True)


def resnet101_atrous(pretrained=True, os=16, **kwargs):
    """Constructs an atrous ResNet-101 model."""
    return _resnet(arch="resnet101", block=Bottleneck, layers=[3, 4, 23, 3],
                   atrous=[2, 2, 2], os=os, pretrained=pretrained,
                   progress=True)


def resnet152_atrous(pretrained=True, os=16, **kwargs):
    """Constructs an atrous ResNet-152 model."""
    return _resnet(arch="resnet152", block=Bottleneck, layers=[3, 8, 36, 3],
                   atrous=[1, 2, 1], os=os, pretrained=pretrained,
                   progress=True)


if __name__ == "__main__":
    torch.manual_seed(0)

    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # torch.cuda.set_device(int(cuda))
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())
    batch, h, w = 4, 256, 256
    x = torch.randn(batch, 3, h, w)
    x = x.to(DEVICE)
    for pretrained in [False, True]:
        for name in __all__:
            announce_msg('testing {}. pretrained: {}'.format(name, pretrained))
            model = sys.modules["__main__"].__dict__[name](
                pretrained=pretrained)
            model.to(DEVICE)
            t0 = dt.datetime.now()
            out = model(x)
            print(
                "in-shape {} \t output-low-shape {} \t output-high-shape {}"
                ". Forward time {}.".format(x.shape, out[0].shape,
                                            out[1].shape,
                                            dt.datetime.now() - t0))
            print("NBR-PARAMS: ", model.get_nbr_params())
            print("Model: {}".format(model))
