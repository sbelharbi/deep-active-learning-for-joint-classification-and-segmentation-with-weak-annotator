#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""criteria.py
Implements different models for task:
- classification only.
- segmentation (and classification)
"""

# All credits of Synchronized BN go to Tamaki Kojima(tamakoji@gmail.com)
# (https://github.com/tamakoji/pytorch-syncbn)
# DeeplabV3:  L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam.  Re-
# thinking  atrous  convolution  for  semantic  image  segmenta-
# tion. arXiv preprint arXiv:1706.05587, 2017..

# Source based: https://github.com/speedinghzl/pytorch-segmentation-toolbox
# BN: https://github.com/mapillary/inplace_abn

# PSPNet:  H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia.  Pyramid scene
# parsing network. In CVPR, pages 2881–2890, 2017.
# https://arxiv.org/abs/1612.01105


import threading
import sys
import math
import os
import datetime as dt
from collections import OrderedDict


from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

sys.path.append("..")

from shared import check_if_allow_multgpu_mode, announce_msg
# sys.path.append("..")
from deeplearning.decision_pooling import WildCatPoolDecision, ClassWisePooling
# lock for threads to protect the instruction that cause randomness and make
# them
thread_lock = threading.Lock()
# thread-safe.

import reproducibility
import constants

ACTIVATE_SYNC_BN = True
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

__all__ = ['lenet5', 'sota_ssl']

# code for CNN13 from
# https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/
# architectures.py


class _BasicNet(nn.Module):
    """
    Basic class for a network.
    """

    def get_nb_params(self):
        """
        Count the number of parameters within the model.
        :return: int, number of learnable parameters.
        """
        return sum([p.numel() for p in self.parameters()])


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
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


class WildCatClassifierHead(nn.Module):
    """
    A WILDCAT type classifier head.
    `WILDCAT: Weakly Supervised Learning of Deep ConvNets for
    Image Classification, Pointwise Localization and Segmentation`,
    Thibaut Durand, Taylor Mordan, Nicolas Thome, Matthieu Cord.
    """
    def __init__(self, inplans, modalities, num_classes, kmax=0.5, kmin=None,
                 alpha=0.6, dropout=0.0):
        super(WildCatClassifierHead, self).__init__()

        self.num_classes = num_classes

        self.to_modalities = weight_norm(nn.Conv2d(
            inplans, num_classes * modalities, kernel_size=1, bias=True))
        self.to_maps = ClassWisePooling(num_classes, modalities)
        self.wildcat = WildCatPoolDecision(
            kmax=kmax, kmin=kmin, alpha=alpha, dropout=dropout)

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        The forward function.
        :param x: input tensor.
        :param seed:
        :param prngs_cuda:
        :return:
        """
        modalities = self.to_modalities(x)
        maps = self.to_maps(modalities)
        scores = self.wildcat(x=maps, seed=seed, prngs_cuda=prngs_cuda)

        return scores


# ==============================================================================
#                         CLASSIFICATION
# ==============================================================================

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes,
                 namenet="ResNet", modalities=4, kmax=0.5, kmin=None,
                 alpha=0.6, dropout=0.0):
        """
        Init. function.
        :param block: class of the block.
        :param layers: list of int, number of layers per block.
        :param num_classes: int, number of output classes. must be > 1.
        ============================= WILDCAT ==================================
        :param modalities: int, number of modalities for WILDCAT.
        :param kmax: int or float scalar in ]0., 1.]. The number of maximum
        features to consider.
        :param kmin: int or float scalar. If None, it takes the same value as
        :param kmax. The number of minimal features to consider.
        :param alpha: float scalar. A weight , used to compute the final score.
        :param dropout: float scalar. If not zero, a dropout is performed over
        the min and max selected features.
        """
        assert num_classes > 1, "Number of classes must be > 1 ....[NOT OK]"
        self.num_classes = num_classes
        self.namenet = namenet

        self.inplanes = 128
        super(ResNet, self).__init__()

        # Encoder

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Find out the size of the output.

        if isinstance(self.layer4[-1], Bottleneck):
            in_channel4 = self.layer1[-1].bn3.weight.size()[0]
            in_channel8 = self.layer2[-1].bn3.weight.size()[0]
            in_channel16 = self.layer3[-1].bn3.weight.size()[0]
            in_channel32 = self.layer4[-1].bn3.weight.size()[0]
        elif isinstance(self.layer4[-1], BasicBlock):
            in_channel4 = self.layer1[-1].bn2.weight.size()[0]
            in_channel8 = self.layer2[-1].bn2.weight.size()[0]
            in_channel16 = self.layer3[-1].bn2.weight.size()[0]
            in_channel32 = self.layer4[-1].bn2.weight.size()[0]
        else:
            raise ValueError("Supported class .... [NOT OK]")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.poolscores = WildCatClassifierHead(
            in_channel32, modalities, num_classes, kmax=kmax, kmin=kmin,
            alpha=alpha, dropout=dropout)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        Forward function.
        :param x: input.
        :param seed: int, a seed for the case of Multigpus to guarantee
        reproducibility for a fixed number of GPUs.
        See  https://discuss.pytorch.org/t/
        did-anyone-succeed-to-reproduce-their-
        code-when-using-multigpus/47079?u=sbelharbi
        In the case of one GPU, the seed in not necessary
        (and it will not be used); so you can set it to None.
        :param prngs_cuda: value returned by torch.cuda.get_prng_state().
        :return:
        """
        # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1.
        x = self.relu1(self.bn1(self.conv1(x)))
        # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1.
        x = self.relu2(self.bn2(self.conv2(x)))
        # 1 / 2: [2, 128, 240, 240]  --> x2^1 to get back to 1.
        x = self.relu3(self.bn3(self.conv3(x)))
        # 1 / 4:  [2, 128, 120, 120] --> x2^2 to get back to 1.
        x = self.maxpool(x)
        # 1 / 4:  [2, 64/256/--, 120, 120]   --> x2^2 to get back to 1.
        x_4 = self.layer1(x)
        # 1 / 8:  [2, 128/512/--, 60, 60]    --> x2^3 to get back to 1.
        x_8 = self.layer2(x_4)
        # 1 / 16: [2, 256/1024/--, 30, 30]   --> x2^4 to get back to 1.
        x_16 = self.layer3(x_8)
        # 1 / 32: [n, 512/2048/--, 15, 15]   --> x2^5 to get back to 1.
        x_32 = self.layer4(x_16)

        # classifier at 32.
        scores32, maps32 = self.poolscores(
            x=x_32, seed=seed, prngs_cuda=prngs_cuda)

        return scores32, maps32

    def get_nb_params(self):
        """
        Count the number of parameters within the model.

        :return: int, number of learnable parameters.
        """
        return sum([p.numel() for p in self.parameters()])

    def __str__(self):
        return "{}(): deep module.".format(
                self.__class__.__name__)


class LeNet5(_BasicNet):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    relu
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    relu
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    relu
    F7 - x (Output)
    Credit: https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py
    """
    def __init__(self, num_classes,
                 namenet="LeNet5"):
        super(LeNet5, self).__init__()

        assert num_classes > 1, "Number of classes must be > 1 ....[NOT OK]"
        self.num_classes = num_classes
        self.namenet = namenet

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, self.num_classes)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x, seed=None, prngs_cuda=None):
        output = self.convnet(x)
        output = output.view(x.size(0), -1)
        output = self.fc(output)
        return output

    def __str__(self):
        return "{}(): LeNet5.".format(self.__class__.__name__)


class SotaSSL(_BasicNet):
    """
    Class of the model used for semi-supervised papers. We have to use this
    one for a fair comparison.
    Original number of parameters: 3 125 908.

    [1]: `S.  Laine  and  T.  Aila.Temporal  ensembling  for  semi-supervised
          learning.CoRR, abs/1610.02242,  2016.`
    [2]: `S. Qiao, W. Shen, Z. Zhang, B. Wang, and A. Yuille.
          Deep co-training for semi-supervised image recognition. InECCV,2018.`
    """

    def __init__(self, num_classes,  dropoutnetssl=0.5, modalities=5, kmax=0.5,
                 kmin=None, alpha=0.6, dropout=0.0):
        super(SotaSSL, self).__init__()

        # define the network.
        # self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(dropoutnetssl)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(dropoutnetssl)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)

        # scores pooling
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        self.fc1 = weight_norm(nn.Linear(128, num_classes))

        # self.poolscores = WildCatClassifierHead(
        #     128, modalities, num_classes, kmax=kmax, kmin=kmin,
        #     alpha=alpha, dropout=dropout)

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        Forward function.
        """
        out = x
        # layer 1-a###
        out = self.conv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)

        # layer 1-b###
        out = self.conv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)

        # layer 1-c###
        out = self.conv1c(out)
        out = self.bn1c(out)
        out = self.activation(out)

        out = self.mp1(out)
        out = self.drop1(out)

        # layer 2-a###
        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)

        # layer 2-b###
        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)

        # layer 2-c###
        out = self.conv2c(out)
        out = self.bn2c(out)
        out = self.activation(out)

        out = self.mp2(out)
        out = self.drop2(out)

        # layer 3-a###
        out = self.conv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)

        # layer 3-b###
        out = self.conv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)

        # layer 3-c###
        out = self.conv3c(out)
        out = self.bn3c(out)
        out = self.activation(out)

        # pool classes' scores.
        out = self.ap3(out)
        out = out.view(-1, 128)
        out = self.fc1(out)

        return out

    def __str__(self):
        return "{}(): SOTA-SSL model.".format(self.__class__.__name__)


# ==============================================================================
#                         SEGMENTATION + CLASSIFICATION
# ==============================================================================

# main ref: https://github.com/VainF/DeepLabV3Plus-Pytorch


def load_url(url, model_dir='./pretrained', map_location=torch.device('cpu')):
    """
    Download pre-trained models.
    :param url: str, url of the pre-trained model.
    :param model_dir: str, path to the temporary folder where the pre-trained
    models will be saved.
    :param map_location: a function, torch.device, string, or dict specifying
    how to remap storage locations.
    :return: torch.load() output. Loaded dict state.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def sota_ssl(**kwargs):
    """
    Constructs a SOTA-SSL model.
    """
    return SotaSSL(**kwargs)


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], namenet="ResNet18", **kwargs)


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], namenet="ResNet50", **kwargs)


def resnet101(**kwargs):
    """
    Constructs a ResNet-101 model.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], namenet="ResNet101", **kwargs)


def lenet5(**kwargs):
    """
    Constructs a LeNet5 network.
    """
    return LeNet5(namenet='LeNet5', **kwargs)


def test_speed():
    import numpy as np
    import random

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_classes = 20
    batch = 200
    poisson = False
    tau = 1.
    model = resnet18(pretrained=False, num_classes=num_classes, dropout=0.5,
                     poisson=poisson, tau=tau)
    model.train()
    print("Num. parameters: {}".format(model.get_nb_params()))
    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # torch.cuda.set_device(int(cuda))
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)
    x = [torch.randn(1, 3, 200, 200) for _ in range(batch)]
    x = [t.to(DEVICE) for t in x]
    labels = torch.randint(low=0, high=num_classes, size=(batch,),
                           dtype=torch.long
                           ).to(DEVICE)
    t0 = dt.datetime.now()
    model(x)
    print("Forwarding a list took {}".format(dt.datetime.now() - t0))

    t0 = dt.datetime.now()
    for xx in x:
        model(xx)
    print("Looping outside the forward()  took {}".format(dt.datetime.now(
    ) - t0))

    x = torch.randn(batch, 3, 200, 200)
    x = x.to(DEVICE)

    t0 = dt.datetime.now()
    model(x)
    print("Forwarding a tensor took {}".format(dt.datetime.now() - t0))


def test_lenet5():
    num_classes = 20
    batch = 3
    model = lenet5(num_classes=num_classes)

    model.train()
    print("Num. parameters: {}".format(model.get_nb_params()))
    cuda = "1"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # torch.cuda.set_device(int(cuda))
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)
    # why 32x32 and not 28x28 mnist?
    # https://github.com/astorfi/TensorFlow-World/tree/master/docs/tutorials/
    # 3-neural_network/autoencoder
    # https://github.com/torch/tutorials/issues/48

    x = torch.randn(batch, 1, 32, 32)
    x = x.to(DEVICE)
    t0 = dt.datetime.now()
    scores = model(x)
    print("Time forward {} of {} samples".format(dt.datetime.now() - t0, batch))
    print(scores.shape, batch)


def test_sota_ssl():
    num_classes = 10
    batch = 3
    model = sota_ssl(num_classes=num_classes, dropoutnetssl=0.5, modalities=5,
                     kmax=0.5, kmin=None, alpha=0.6, dropout=0.0)

    model.train()
    print("Num. parameters: {}".format(model.get_nb_params()))
    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # torch.cuda.set_device(int(cuda))
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)
    # why 32x32 and not 28x28 mnist?
    # https://github.com/astorfi/TensorFlow-World/tree/master/docs/tutorials/
    # 3-neural_network/autoencoder
    # https://github.com/torch/tutorials/issues/48

    x = torch.randn(batch, 3, 32, 32)
    x = x.to(DEVICE)
    t0 = dt.datetime.now()
    scores = model(x)
    print("Time forward {} of {} samples".format(dt.datetime.now() - t0, batch))
    print(scores.shape, batch)


if __name__ == "__main__":
    torch.manual_seed(0)
    # test_sota()
    # test_lenet5()
    # test_sota_ssl()
