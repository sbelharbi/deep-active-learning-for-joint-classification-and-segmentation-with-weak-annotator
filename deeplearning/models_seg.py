#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements models that can perform both tasks:
- classification.
- segmentation.
"""
# main ref: https://github.com/VainF/DeepLabV3Plus-Pytorch

import threading
import sys
import math
import os
import datetime as dt
from collections import OrderedDict
import itertools


from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter

sys.path.append("..")

from shared import check_if_allow_multgpu_mode, announce_msg

from deeplearning.deeplab import deeplab_v3_plus_head
from deeplearning.backbone import resnet
from deeplearning.backbone import xception
from deeplearning.wildcat import WildCatClassifierHead


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

from deeplearning.syncbn import nn as NN_Sync_BN

if check_if_allow_multgpu_mode() and ACTIVATE_SYNC_BN:  # Activate Synch-BN.
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

__all__ = ['hybrid_model']


def convrelu(in_channels, out_channels, kernel, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel,
                  stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    """
    Build twice the module:
    Conv2v -> Batchnorm2d -> Relu
    """

    def __init__(self, in_channels, out_channels, kernel=3, stride=1,
                 padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                      padding=padding,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class HybridModel(nn.Module):
    """
    Model that performs: classification and segmentation.
    U-NET style: encoder and decoder based architecture.
    The encoder is Resnet-based.
    """
    def __init__(self,
                 num_classes,
                 num_masks=1,
                 backbone=constants.RESNET18,
                 pretrained=True,
                 modalities=4,
                 kmax=0.5,
                 kmin=None,
                 alpha=0.6,
                 dropout=0.0,
                 backbone_dropout=0.0,
                 freeze_classifier=False,
                 base_width=16,
                 leak=64
                 ):
        """
        Init. function.
        :param num_classes: int. number of classes for classification.
        :param num_masks: int. number of masks to be produced. supported: 1.
        :param backbone: str. name of the backbone. see constants.py
        :param pretrained: bool. if true, the backbone loaded with imagenet
        pretrained weights.
        :param modalities: int, number of modalities for WILDCAT.
        :param kmax: int or float scalar in ]0., 1.]. The number of maximum
        features to consider.
        :param kmin: int or float scalar. If None, it takes the same value as
        :param kmax. The number of minimal features to consider.
        :param alpha: float scalar. A weight , used to compute the final score.
        :param dropout: float scalar. If not zero, a dropout is performed over
        the min and max selected features. (wildcat)
        :param backbone_dropout: float scalar. dropout that is used within
        the entire net for bayesian inference.
        :param freeze_classifier: bool. whether to freeze or not the classifier.
        :param base_width: int. base width that increases the complexity of
        the upscale part of U-Net. The higher this value, the higher the
        number of parameters of the upscale-part of U-Net.
        :param leak: int. number of features maps to be extracted from the
        classifier maps to be used for segmentation.
        """
        super(HybridModel, self).__init__()
        self.name = "hybrid_model"
        self.freeze_classifier = freeze_classifier

        msg = "backbone `{}` unsupported. see constants.backbones {}.".format(
            backbone, constants.backbones
        )
        assert backbone in constants.backbones, msg
        msg = "supported 'num_masks'=1. provided {}.".format(num_masks)
        assert num_masks == 1, msg

        if backbone in constants.resnet_backbones:
            self.backbone = resnet.__dict__[backbone](pretrained=pretrained)
        elif 'xception' in backbone:
            raise NotImplementedError
        else:
            raise ValueError('How did you end up here?')

        msg = "'base_width' must be int. found {}.".format(type(base_width))
        assert isinstance(base_width, int), msg

        msg = "'base_width' must be > 0. found {}.".format(base_width)
        assert base_width > 0, msg

        msg = "'leak' must be int. found {}.".format(type(base_width))
        assert isinstance(leak, int), msg
        msg = "'leak' must be > 0. found {}.".format(leak)
        assert leak > 0, msg


        self.backbone_dropout = backbone_dropout
        self.dropoutlayer = nn.Dropout(p=backbone_dropout)

        self.layer0 = nn.Sequential(self.backbone.conv1,
                                    self.backbone.bn1,
                                    self.backbone.relu)
        self.layer1 = nn.Sequential(self.backbone.maxpool,
                                    *self.backbone.layer1)
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        self.dimout0 = self.backbone.dimout0
        self.dimout1 = self.backbone.dimout1
        self.dimout2 = self.backbone.dimout2
        self.dimout3 = self.backbone.dimout3
        self.dimout4 = self.backbone.dimout4

        self.b_w = base_width

        self.lk = leak

        # 1x1 conv: bridge
        self.layer0_1x1 = convrelu(self.dimout0, self.lk, 3, 1, 0)
        self.layer1_1x1 = convrelu(self.dimout1, self.lk, 3, 1, 0)
        self.layer2_1x1 = convrelu(self.dimout2, self.lk, 3, 1, 0)
        self.layer3_1x1 = convrelu(self.dimout3, self.lk, 3, 1, 0)
        self.layer4_1x1 = convrelu(self.dimout4, self.lk, 3, 1, 0)

        # parallel down path for segmentation.
        self.down0 = DoubleConv(64 + self.lk, self.b_w * 2, 3, 2, 1)
        self.down1 = DoubleConv(self.b_w * 2 + self.lk, self.b_w * 4, 3, 2, 1)
        self.down2 = DoubleConv(self.b_w * 4 + self.lk, self.b_w * 8, 3, 2, 1)
        self.down3 = DoubleConv(self.b_w * 8 + self.lk, self.b_w * 16, 3, 2, 1)
        self.down4 = DoubleConv(self.b_w * 16 + self.lk, self.b_w * 32, 3, 2, 1)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True
                                    )

        ft = 2

        self.conv_up3 = convrelu(self.b_w * 32 + self.b_w * 16, 512 // ft, 3,
                                 1, 1)
        self.conv_up2 = convrelu(self.b_w * 8 + 512 // ft, 256 // ft, 3, 1, 1)
        self.conv_up1 = convrelu(self.b_w * 4 + 256 // ft, 256 // ft, 3, 1, 1)
        self.conv_up0 = convrelu(self.b_w * 2 + 256 // ft, 128 // ft, 3, 1, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1, 1)  # down-path starts

        self.conv_original_size2 = convrelu(64 + 128 // ft, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, num_masks, 1)

        # classification head.
        self.classifier = WildCatClassifierHead(self.dimout4,
                                                modalities,
                                                num_classes,
                                                kmax=kmax,
                                                kmin=kmin,
                                                alpha=alpha,
                                                dropout=dropout
                                                )

    def train(self, mode=True):
        """
        Override nn.Module.train() to consider the frozen classifier.
        """
        super(HybridModel, self).train(mode=mode)

        if self.freeze_classifier:
            self.freeze_cl()   # call freeze to fix the batch-norm.

        return self

    def freeze_cl(self):
        """
        Freeze the classifier.
        Including the batchnorm.
        This function is to be called from outside explicitly.
        """
        msg = "'self.freeze_classifier' is False while you asked to " \
              "freeze the classifier. "
        assert self.freeze_classifier, msg

        for mdl in [self.layer0, self.layer1, self.layer2, self.layer3,
                    self.layer4, self.classifier]:
            for param in mdl.parameters():
                param.requires_grad = False  # discard from update.

            for module in mdl.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(
                        module, NN_Sync_BN.BatchNorm2d):
                    module.eval()  # turn off updating the running avg, var.

                if isinstance(module, nn.Dropout):
                    module.eval()  # turn off dropout.

    def unfreeze_cl(self):
        """
        Un-freeze the classifier.
        This function is to be called from outside explicitly.
        :return:
        """

        self.freeze_classifier = False  # turn off the freezing variable.

        for mdl in [self.layer0, self.layer1, self.layer2, self.layer3,
                    self.layer4, self.classifier]:
            for param in mdl.parameters():
                param.requires_grad = True  # turn on tracking gradients.

            for module in mdl.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(
                        module, NN_Sync_BN.BatchNorm2d):
                    module.train(mode=True)  # turn on updating the running
                    # avg, var.

                if isinstance(module, nn.Dropout):
                    module.train(mode=True)  # turn on dropout.

    def assert_cl_is_frozen(self):
        """
        Check that all the classifier is frozen.
        :return:
        """
        msg = "'self.freeze_classifier' is False while you asked to " \
              "check if the classifier is frozen. "
        assert self.freeze_classifier, msg


        for mdl in [self.layer0, self.layer1, self.layer2, self.layer3,
                    self.layer4, self.classifier]:
            for param in mdl.parameters():
                msg = "Found a parameter with `requires_grad` set to True. " \
                      "It should be False."
                assert not param.requires_grad, msg

            for module in mdl.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(
                        module, NN_Sync_BN.BatchNorm2d):
                    msg = "'.training' is True. Expected False. " \
                          "BachNorm."
                    assert not module.training, msg

                if isinstance(module, nn.Dropout):
                    msg = "'.training' is True. Expected False. " \
                          "Dropout."
                    assert not module.training, msg

        return True

    def print_cl(self):
        """
        Printout the classifier.
        :return:
        """
        for mdl in [self.layer0, self.layer1, self.layer2, self.layer3,
                    self.layer4, self.classifier]:
            print(repr(mdl))

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        Forward function.
        :param x: input image tensor.
        :return: scores: raw scores for global classification.
                 masks: raw scores of segmentation (unnormalized).
        """
        msg = "'x.ndim' must be 4 (bsz, nbr_plans, h, w). found {}.".format(
            x.ndim
        )
        assert x.ndim == 4, msg

        img_shape = x.shape
        h, w = x.shape[2], x.shape[3]

        x_original0 = self.conv_original_size0(x)
        x_original1 = self.conv_original_size1(x_original0)

        layer0 = self.layer0(x)

        if self.backbone_dropout > 0:
            layer0 = self.dropoutlayer(layer0)

        layer1 = self.layer1(layer0)

        if self.backbone_dropout > 0:
            layer1 = self.dropoutlayer(layer1)

        layer2 = self.layer2(layer1)

        if self.backbone_dropout > 0:
            layer2 = self.dropoutlayer(layer2)

        layer3 = self.layer3(layer2)

        if self.backbone_dropout > 0:
            layer3 = self.dropoutlayer(layer3)

        layer4 = self.layer4(layer3)

        if self.backbone_dropout > 0:
            layer4 = self.dropoutlayer(layer4)

        # ======================================================================
        #                              CLASSIFICATION

        scores, maps = self.classifier(layer4)
        # resize maps to the same size as input x
        if maps.shape[2:] != img_shape[2:]:
            maps = F.interpolate(input=maps,
                                 size=(h, w),
                                 mode='bilinear',
                                 align_corners=True
                                 )


        # ======================================================================

        # ======================================================================

        # ======================================================================
        #                              SEGMENTATION

        # classification down-leak
        layer4 = self.layer4_1x1(layer4)
        layer3 = self.layer3_1x1(layer3)
        layer2 = self.layer2_1x1(layer2)
        layer1 = self.layer1_1x1(layer1)
        layer0 = self.layer0_1x1(layer0)

        # down-path ============================================================
        # down 0.
        tmp_x_1 = x_original1
        if x_original1.shape[2:] != layer0.shape[2:]:
            tmp_x_1 = F.interpolate(input=x_original1,
                                   size=(layer0.shape[2],
                                         layer0.shape[3]),
                                   mode='bilinear', align_corners=True)

        down0 = self.down0(torch.cat([tmp_x_1, layer0], dim=1))
        if self.backbone_dropout > 0:
            down0 = self.dropoutlayer(down0)

        # down 1
        if down0.shape[2:] != layer1.shape[2:]:
            down0 = F.interpolate(input=down0,
                                   size=(layer1.shape[2],
                                         layer1.shape[3]),
                                   mode='bilinear', align_corners=True)


        down1 = self.down1(torch.cat([down0, layer1], dim=1))
        if self.backbone_dropout > 0:
            down1 = self.dropoutlayer(down1)

        # down 2
        if down1.shape[2:] != layer2.shape[2:]:
            down1 = F.interpolate(input=down1,
                                   size=(layer2.shape[2],
                                         layer2.shape[3]),
                                   mode='bilinear', align_corners=True)

        down2 = self.down2(torch.cat([down1, layer2], dim=1))
        if self.backbone_dropout > 0:
            down2 = self.dropoutlayer(down2)

        # down 3
        if down2.shape[2:] != layer3.shape[2:]:
            down2 = F.interpolate(input=down2,
                                   size=(layer3.shape[2],
                                         layer3.shape[3]),
                                   mode='bilinear', align_corners=True)

        down3 = self.down3(torch.cat([down2, layer3], dim=1))
        if self.backbone_dropout > 0:
            down3 = self.dropoutlayer(down3)

        # down 4
        if down3.shape[2:] != layer4.shape[2:]:
            down3 = F.interpolate(input=down3,
                                   size=(layer4.shape[2],
                                         layer4.shape[3]),
                                   mode='bilinear', align_corners=True)

        down4 = self.down4(torch.cat([down3, layer4], dim=1))
        if self.backbone_dropout > 0:
            down4 = self.dropoutlayer(down4)

        # up-path ==============================================================
        x = self.upsample(down4)

        # readjust the size
        if down3.shape[2:] != x.shape[2:]:
            x = F.interpolate(input=x, size=(down3.shape[2], down3.shape[3]),
                              mode='bilinear', align_corners=True)
        x = torch.cat([x, down3], dim=1)
        x = self.conv_up3(x)

        if self.backbone_dropout > 0:
            x = self.dropoutlayer(x)

        x = self.upsample(x)

        # readjust the size
        if down2.shape[2:] != x.shape[2:]:
            x = F.interpolate(input=x, size=(down2.shape[2], down2.shape[3]),
                              mode='bilinear', align_corners=True)
        x = torch.cat([x, down2], dim=1)
        x = self.conv_up2(x)

        if self.backbone_dropout > 0:
            x = self.dropoutlayer(x)

        x = self.upsample(x)

        # readjust the size
        if down1.shape[2:] != x.shape[2:]:
            x = F.interpolate(input=x, size=(down1.shape[2], down1.shape[3]),
                              mode='bilinear', align_corners=True)
        x = torch.cat([x, down1], dim=1)
        x = self.conv_up1(x)

        if self.backbone_dropout > 0:
            x = self.dropoutlayer(x)

        x = self.upsample(x)

        # readjust the size
        if down0.shape[2:] != x.shape[2:]:
            x = F.interpolate(input=x, size=(down0.shape[2], down0.shape[3]),
                              mode='bilinear', align_corners=True)

        x = torch.cat([x, down0], dim=1)
        x = self.conv_up0(x)

        if self.backbone_dropout > 0:
            x = self.dropoutlayer(x)

        x = self.upsample(x)
        # readjust the size
        if x.shape[2:] != img_shape[2:]:
            x = F.interpolate(input=x,
                              size=(h, w),
                              mode='bilinear',
                              align_corners=True
                              )

        x = torch.cat([x, x_original1], dim=1)
        x = self.conv_original_size2(x)

        masks = self.conv_last(x)

        msg = "h/w mismatch: img {}, mask {}.".format(
            img_shape[2:], masks.shape[2:])
        assert img_shape[2:] == masks.shape[2:], msg

        # ======================================================================

        return scores, masks, maps

    def set_dropout_to_train_mode(self):
        """
        For dropout module into train mode. useful for bayesian dropout.
        use with caution.
        :return:
        """
        self.dropoutlayer.train(mode=True)

    def set_dropout_to_eval_mode(self):
        """
        For dropout module into eval mode. useful for bayesian dropout.
        use with caution.
        :return:
        """
        self.dropoutlayer.eval()

    def get_nbr_params(self):
        """
        Compute the number of parameters of the model.
        :return:
        """
        return sum([p.numel() for p in self.parameters()])

    def __str__(self):
        return "Model: {}. Backbone: {}. Classifier: {}. " \
               "Freeze classifier: {}.".format(
            self.name,
            self.backbone.name,
            self.classifier.name,
            self.freeze_classifier
        )


def hybrid_model(num_classes,
                 num_masks=1,
                 backbone=constants.RESNET18,
                 pretrained=True,
                 modalities=4,
                 kmax=0.5,
                 kmin=None,
                 alpha=0.6,
                 dropout=0.0,
                 backbone_dropout=0.0,
                 freeze_classifier=False,
                 base_width=24,
                 leak=64
                 ):
    return HybridModel(num_classes=num_classes,
                       num_masks=num_masks,
                       backbone=backbone,
                       pretrained=pretrained,
                       modalities=modalities,
                       kmax=kmax,
                       kmin=kmin,
                       alpha=alpha,
                       dropout=dropout,
                       backbone_dropout=backbone_dropout,
                       freeze_classifier=freeze_classifier,
                       base_width=base_width,
                       leak=leak
                       )


def load_url(url,
             model_dir='./pretrained',
             map_location=torch.device('cpu')
             ):
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


def test_HybridModel_freeze_cl():
    """
    Test `HybridModel`.
    :return:
    """
    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # torch.cuda.set_device(int(cuda))
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())

    num_classes = 10
    num_masks = 1
    batch = 3
    base_width = 16
    x = torch.randn(batch, 3, 256, 253)
    x = x.to(DEVICE)
    for backbone_dropout in [0., 0.1]:
        for pretrained in [False, True]:
            for freeze_cl in [False, True]:
                for backbone in constants.backbones:
                    model = hybrid_model(num_classes, num_masks=num_masks,
                                         backbone=backbone,
                                         pretrained=pretrained,
                                         modalities=4, kmax=0.5, kmin=None,
                                         alpha=0.6, dropout=0.0,
                                         backbone_dropout=backbone_dropout,
                                         freeze_classifier=freeze_cl,
                                         base_width=base_width
                                         )

                    model.eval()
                    nbr_params_total = model.get_nbr_params()
                    cl_params = model.classifier.get_nbr_params()
                    announce_msg("{} (pret: {}). "
                                 "Total-params: {}. "
                                 "CL-params: {} ({:.2f}%) ".format(
                                    backbone,
                                    pretrained,
                                    nbr_params_total,
                                    cl_params,
                                    100. * cl_params / float(nbr_params_total),
                                    )
                                 )
                    # DEVICE = torch.device("cpu")
                    model.to(DEVICE)
                    t0 = dt.datetime.now()
                    scores, masks, maps = model(x)
                    print("Time forward {} of {} samples".format(
                        dt.datetime.now() - t0, batch))
                    print(
                        "in: ", x.shape, "scores: ", scores.shape, " masks: ",
                        masks.shape)
                    print("Model: {}".format(model))
                    print("Min-max output-masks: {}, {}".format(
                        masks.min(), masks.max()))
                    msg = "h, w mismatch: x {}  masks {}".format(
                        x.shape[2:], masks[2:])
                    assert x.shape[2:] == masks.shape[2:], msg

                    if backbone_dropout != 0.:
                        model.set_dropout_to_train_mode()
                    print("backbone dropout: {}".format(backbone_dropout))
                    scores, masks = model(x)
                    print("Scores first forward:")
                    print(scores)
                    scores, masks = model(x)
                    print("Scores second forward:")
                    print(scores)

                    # test freezing the classifier
                    if freeze_cl:
                        print("going to freeze the classifier:")
                        model.freeze_cl()
                        scores, masks = model(x)
                        print("Scores after freezing the classifier:")
                        print(scores)


def test_HybridModel():
    """
    Test `HybridModel`.
    :return:
    """
    cuda = "6"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # torch.cuda.set_device(int(cuda))
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())

    num_classes = 10
    num_masks = 1
    batch = 3
    x = torch.randn(batch, 3, 256, 253)
    x = x.to(DEVICE)
    for backbone_dropout in [0., 0.1]:
        for pretrained in [False, True]:
            for freeze_cl in [False, True]:
                for backbone in constants.backbones:
                    model = hybrid_model(num_classes, num_masks=num_masks,
                                         backbone=backbone,
                                         pretrained=pretrained,
                                         modalities=4, kmax=0.5, kmin=None,
                                         alpha=0.6, dropout=0.0,
                                         backbone_dropout=backbone_dropout,
                                         freeze_classifier=freeze_cl
                                         )
                    model.eval()
                    nbr_params_total = model.get_nbr_params()
                    cl_params = model.classifier.get_nbr_params()
                    announce_msg("{} (pret: {}). "
                                 "Total-params: {}. "
                                 "CL-params: {} ({:.2f}%) ".format(
                                    backbone,
                                    pretrained,
                                    nbr_params_total,
                                    cl_params,
                                    100. * cl_params / float(nbr_params_total),
                                    )
                                 )
                    # DEVICE = torch.device("cpu")
                    model.to(DEVICE)
                    t0 = dt.datetime.now()
                    scores, masks, maps = model(x)
                    print("Time forward {} of {} samples".format(
                        dt.datetime.now() - t0, batch))
                    print(
                        "in: ", x.shape,
                        "scores: ", scores.shape,
                        "masks: ", masks.shape,
                        "maps: ", maps.shape
                    )
                    print("Model: {}".format(model))
                    print("Min-max output-masks: {}, {}".format(
                        masks.min(), masks.max()))
                    msg = "h, w mismatch: x {}  masks {}".format(
                        x.shape[2:], masks[2:])
                    assert x.shape[2:] == masks.shape[2:], msg

                    if backbone_dropout != 0.:
                        model.set_dropout_to_train_mode()
                    print("backbone dropout: {}".format(backbone_dropout))
                    scores, masks, maps = model(x)
                    print("Scores first forward:")
                    print(scores)
                    scores, masks, maps = model(x)
                    print("Scores second forward:")
                    print(scores)

                    # test freezing the classifier
                    if freeze_cl:
                        print("going to freeze the classifier:")
                        model.freeze_cl()
                        scores, masks, maps = model(x)
                        print("Scores after freezing the classifier:")
                        print(scores)



if __name__ == "__main__":
    torch.manual_seed(0)
    # test_sota()
    # test_lenet5()
    test_HybridModel()
