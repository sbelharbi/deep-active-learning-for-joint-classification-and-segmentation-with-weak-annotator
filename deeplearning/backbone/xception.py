"""
Main refs:
https://github.com/YudeWang/deeplabv3plus-pytorch
https://github.com/VainF/DeepLabV3Plus-Pytorch
https://github.com/yassouali/pytorch_segmentation
"""
import sys
import os
import datetime as dt
import math

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F


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

bn_mom = 0.0003

__all__ = ['xception']

model_urls = {
    'xception': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
    # 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
    # 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                                  bias=False)
            self.skipbn = BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        self.relu = nn.ReLU(inplace=True)

        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1,
                                   dilation=dilation))
        rep.append(BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1,
                                   dilation=dilation))
        rep.append(BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=stride,
                                   dilation=dilation))
        rep.append(BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [
                self.relu,
                SeparableConv2d(in_channels, in_channels, 3, 1, dilation),
                BatchNorm2d(in_channels)]

        if not use_1st_relu: rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        x = output + skip
        return x


class Xception(nn.Module):
    def __init__(self, output_stride=16, in_channels=3, pretrained=True):
        super(Xception, self).__init__()

        self.name = "xception"

        # Stride for block 3 (entry flow), and the dilation rates for middle
        # flow and exit flow
        if output_stride == 16:
            b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8:
            b3_s, mf_d, ef_d = 1, 2, (2, 4)

        # Entry Flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64)

        self.block1 = Block(64, 128, stride=2, dilation=1, use_1st_relu=False)
        self.block2 = Block(128, 256, stride=2, dilation=1)
        self.block3 = Block(256, 728, stride=b3_s, dilation=1)

        # Middle Flow
        for i in range(16):
            exec(
                f'self.block{i + 4} = Block(728, 728, stride=1, dilation=mf_d)')

        # Exit flow
        self.block20 = Block(728, 1024, stride=1, dilation=ef_d[0],
                             exit_flow=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn3 = BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn4 = BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=ef_d[1])
        self.bn5 = BatchNorm2d(2048)

        initialize_weights(self)
        if pretrained:
            self._load_pretrained_model()

    def _load_pretrained_model(self):
        """
        Load imagenet pretrained xception.
        :return:
        """
        pretrained_weights = load_state_dict_from_url(
            url=model_urls['xception'], model_dir='./pretrained-imgnet',
            map_location=torch.device('cpu'), progress=True)
        state_dict = self.state_dict()
        model_dict = {}

        for k, v in pretrained_weights.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)  # [C, C] -> [C, C, 1, 1]
                if k.startswith('block11'):
                    # In Xception there is only 8 blocks in Middle flow
                    model_dict[k] = v
                    for i in range(8):
                        model_dict[k.replace('block11', f'block{i + 12}')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        low_level_features = x
        x = F.relu(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return low_level_features, x

    def get_nbr_params(self):
        """
        Compute the number of parameters of the model.
        :return:
        """
        return sum([p.numel() for p in self.parameters()])

    def __str__(self):
        return "{}: MODIFIED XCEPTION .".format(self.name)


def xception(pretrained=True, output_stride=16):
    return Xception(
        output_stride=output_stride, in_channels=3, pretrained=pretrained)


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
        announce_msg('testing {}. pretrained: {}'.format('xception',
                                                         pretrained))
        model = sys.modules["__main__"].__dict__['xception'](
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
