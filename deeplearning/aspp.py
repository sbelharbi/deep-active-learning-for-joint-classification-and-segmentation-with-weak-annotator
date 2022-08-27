"""
Main refs:
https://github.com/YudeWang/deeplabv3plus-pytorch
https://github.com/VainF/DeepLabV3Plus-Pytorch
"""
import sys
import os
import datetime as dt

import torch
from torch import nn
from torch.nn import functional as F


sys.path.append("..")
sys.path.append("../..")

from shared import check_if_allow_multgpu_mode, announce_msg
from deeplearning.utils import initialize_weights

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

__all__ = ['aspp']


# The Atrous Spatial Pyramid Pooling


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding,
                      dilation=dilation, bias=False),
            BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))


class ASPP(nn.Module):
    """
    Implements Atrous Spatial Pyramid Pooling (ASPP).
    """
    def __init__(self, in_channels, output_stride):
        super(ASPP, self).__init__()

        msg = 'Only output strides of 8 or 16 are supported.'
        assert output_stride in [8, 16], msg
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)),
                           mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

    def get_nbr_params(self):
        """
        Compute the number of parameters of the model.
        :return:
        """
        return sum([p.numel() for p in self.parameters()])


def aspp(in_channels, output_stride):
    return ASPP(in_channels=in_channels, output_stride=output_stride)


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
    x = torch.randn(batch, 3, h, w)
    x = x.to(DEVICE)

    model = aspp(in_channels=3, output_stride=16)
    announce_msg('testing {}.'.format(ASPP))
    model.to(DEVICE)
    t0 = dt.datetime.now()
    out = model(x)
    print(
        "in-shape {} \t output-shape {} "
        ". Forward time {}.".format(x.shape, out.shape,
                                    dt.datetime.now() - t0))
    print("NBR-PARAMS: ", model.get_nbr_params())
