import sys
import os
import datetime as dt

import torch
from torch import nn
from torch.nn import functional as F

sys.path.append("..")

from shared import check_if_allow_multgpu_mode, announce_msg
from deeplearning.utils import initialize_weights
from deeplearning.aspp import aspp

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


__all__ = ["deeplab_v3_plus_head"]


class Decoder(nn.Module):
    """
    Implement a decoder for deeplabv3+.
    """
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        h, w = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


class DeepLabV3PlusHead(nn.Module):
    """
    DeepLabV3+ head.
    """
    def __init__(self, num_classes=1, backbone='xception',
                 output_stride=16, freeze_bn=False):
        """
        Init function.

        :param num_classes: int. number of segmentation masks to output.
        :param backbone: str. name of the backbone.
        :param output_stride: output stride. supported: 8, 16.
        :param freeze_bn: bool. if true, the batchnorm parameters are frozen.
        """
        super(DeepLabV3PlusHead, self).__init__()

        self.name = "deeblabv3plus"
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            low_level_channels = 256
        elif 'xception' in backbone:
            low_level_channels = 128
        else:
            raise ValueError('How did you get here?')

        self.ASSP = aspp(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, low_level_features, x, h, w):
        """
        The forward function.
        :param low_level_features: feature at low level.
        :param x: features at high level.
        :param h: int. original height of the image.
        :param w: int. original width of the image.
        :return: x: row output scores (unnormalized).
        """
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, BatchNorm2d):
                module.eval()

    def get_nbr_params(self):
        """
        Compute the number of parameters of the model.
        :return:
        """
        return sum([p.numel() for p in self.parameters()])

    def __str__(self):
        return "{}: DEEPLABV3+.".format(self.name)


def deeplab_v3_plus_head(
        num_classes=1, backbone='xception', output_stride=16, freeze_bn=False):
    return DeepLabV3PlusHead(
        num_classes=num_classes, backbone=backbone, output_stride=output_stride,
        freeze_bn=freeze_bn)


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
    batch, h, w = 4, 32, 32
    in_channels = 2048
    for backbone in constants.backbones:
        if 'resnet' in backbone:
            low_level_channels = 256
        elif 'xception' in backbone:
            low_level_channels = 128

        x = torch.randn(batch, in_channels, h // 4, w // 4)
        low_level_features = torch.randn(
            batch, low_level_channels, h // 2, w // 2)
        x = x.to(DEVICE)
        low_level_features = low_level_features.to(DEVICE)
        announce_msg('testing {}. backbone {}.'.format(
            DeepLabV3PlusHead, backbone))

        model = deeplab_v3_plus_head(num_classes=1, backbone=backbone,
                                     output_stride=16, freeze_bn=False)
        model.to(DEVICE)
        t0 = dt.datetime.now()
        out = model(low_level_features, x, h, w)
        print(
            "in-shape {} \t output-shape {} "
            ". Forward time {}.".format(x.shape, out.shape,
                                        dt.datetime.now() - t0))
        print("NBR-PARAMS: ", model.get_nbr_params())
        print("Model: {}".format(model))
        print("Min-max output: {}, {}".format(out.min(), out.max()))
