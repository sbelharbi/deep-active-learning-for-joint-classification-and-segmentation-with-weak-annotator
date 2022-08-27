import sys
import os
from os.path import join
import datetime as dt
import math
import numbers
import subprocess
import fnmatch
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms
import tqdm
from PIL import Image
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt

sys.path.append("..")
from reproducibility import set_default_seed
from reproducibility import reset_seed
import constants
from loader import PhotoDataset
from deeplearning.criteria import Dice
from deeplearning.criteria import GetAreaOfMask
from instantiators import instantiate_optimizer
from tools import Dict2Obj
from tools import plot_curves_from_dict
from tools import VisualiseTemporalMask
from shared import csv_loader


DEBUG_SEG = False  # used for debug. Turn this off otherwise.
NORMALIZE_HISTO = True  # wheter to normalize the histogram to probabilities
# or not. HARD CODED


class _LBPConv2D(nn.Module):
    """
    Implement uniform LBP feature extraction using 2d convolution with fixed
    filters.
    NOTE:
        BATCHABLE. NON-LEARNABLE PARAMETERS (fixed).

    Computes LBP feature map for EACH INPUT CHANNEL.
    operates on inputs with 1 single channel. if the input has many channels,
    we reshape it to have one channel only.

    Uniform LBP encodes each pixel using its neighbors using a binary string
    that is converted into an integer code. depending on the number of
    neighbors, assuming it is n, we can obtain in total 2^n code (pattern).
    in this implementation, n depends on the size of the kernel.

    When using histogram to quantify an image encoded using LBP features,
    the bounds of the histogram depends on the number of possible patterns.
    to bound histograms with the SAME bound (e.g. [0., 1.]) independently
    from n, we can normalize the lbp feature at each pixel by the total
    number of possible patterns. this normalization will bound the values into
    [0, 1.[.

    LBP: fixed_weight_1x1_2d(activation(fixed_wieght_conv2d(image))).

    Based on:
    'Local Binary Convolutional Neural Networks'.
    https://arxiv.org/pdf/1608.06049.pdf
    """
    def __init__(self, kernel_size, exact_conv=True,
                 padding_mode='reflect', normalize=False):
        """
        Init.  function.

        :param kernel_size: int. size of the kernel. (squared kernels)
        :param exact_conv: bool. if true, the obtained lbp feature maps have
        the same hieght and width as the input.
        :param padding_mode: str. padding mode: 'reflect', 'replicate', or
        'circular'.
        :param normalize: bool. if true, we normalize the obtained int code
        at each pixel by the total number of possible codes.
        :returns: LBP feature maps: in_channel maps. shape:
        batch_size, c, h`, w`.
        """
        super(_LBPConv2D, self).__init__()

        in_channels = 1  # works only with input with 1 channel. if not the
        # case, we reshape the tensor to be so.
        msg = "'kernel_size' must be odd. found {}.".format(kernel_size)
        assert (kernel_size - 1) % 2 == 0, msg

        msg = "'kernel_size' must be > 1. found {}.".format(kernel_size)
        assert kernel_size > 1, msg

        allowed_padd_modes = ['reflect', 'replicate', 'circular']
        msg = "'padding_mode'={} not allowed. allowed:{}.".format(
            padding_mode, allowed_padd_modes)
        assert padding_mode in allowed_padd_modes, msg

        self.padding_sz = 0
        self.padding_mode = padding_mode
        self.exact_conv = exact_conv
        self.kernel_size = kernel_size
        self.normalize = normalize

        if exact_conv:
            self.padding_sz = int((kernel_size - 1) / 2)
            msg = "you asked for exact convolution. for this, the size of " \
                  "the kernel must be odd. found {}.".format(kernel_size)
            assert (kernel_size - 1) % 2 == 0, msg

        self.nbr_neighbors = kernel_size + kernel_size - 1 + kernel_size - 1 + \
                        kernel_size - 2
        kernel = torch.zeros((self.nbr_neighbors, 1, kernel_size, kernel_size))
        # set the central value to -1.
        kernel[:, :, int(kernel_size/2), int(kernel_size/2)] = -1.

        weight_lbp = (2**torch.arange(0, self.nbr_neighbors)).float().view(
            1, self.nbr_neighbors, 1, 1
        )

        if normalize:
            weight_lbp = weight_lbp / float(2**self.nbr_neighbors)

        kernel = self.set_neighboors_to_1(kernel)

        self.register_buffer("weight2d", kernel)
        self.register_buffer("weight1x1", weight_lbp)

        self.conv2d = F.conv2d
        self.conv1x1 = F.conv2d
        self.activation = torch.sigmoid

    def set_neighboors_to_1(self, filter):
        """
        Set the neighbors into 1.
        :return:
        """
        props = [0, self.kernel_size - 1]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i in props or j in props:
                    filter[:, :, i, j] = 1.
        return filter

    def forward(self, x):
        """
        Forward function.
        :param x: pytorch tensor of shape (batch_size, c, h, w).
        :return: uniform lbp for each channel (batch_size, c, h`, w`).
        if self.exact_conv is true, h`==h, w`=w.
        """
        assert x.ndim == 4, "input.ndim must be 4. found {}.".format(x.ndim)
        bs, c, h, w = x.shape
        input = x.view(bs*c, 1, h, w)  # self.conv2d operates only on inputs
        # with 1 channel.

        if self.exact_conv:
            input = F.pad(input, pad=[self.padding_sz] * 4,
                          mode=self.padding_mode)
        # conv2d
        input = self.conv2d(input, weight=self.weight2d)  # bs*c, nbr_n, h`, w`
        input = self.activation(input)
        input = self.conv1x1(input, weight=self.weight1x1)  # bs*c, 1, h`, w`
        _, _, hp, wp = input.shape

        out = input.view(bs, c, hp, wp)

        if self.exact_conv:
            assert hp == h, "hp={}, h={} mismatch. mode: exact_conv=True."
            assert wp == w, "wp={}, w={} mismatch. mode: exact_conv=True."

        return out


class _LBPModule(nn.Module):
    """
    Implement the histogram of uniform LBP features with multiple scales.

    NOTE:
        BATCHABLE.
    """
    def __init__(self, kernel_sizes, exact_conv=True,
                 padding_mode='reflect', normalize=False):
        """
        Init. function.
        :param kernel_sizes: list of int. size of the kernels. (squared kernels)
        :param exact_conv: bool. if true, the obtained lbp feature maps have
        the same hieght and width as the input.
        :param padding_mode: str. padding mode: 'reflect', 'replicate', or
        'circular'.
        :param normalize: bool. if true, we normalize the obtained int code
        at each pixel by the total number of possible codes.
        :returns: LBP feature maps: in_channel maps. shape:
        batch_size, c, h`, w`.
        """
        super(_LBPModule, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.exact_conv = exact_conv

        msg = "exact_conv must be true to be able to concatenate the lbp " \
              "fetaures in one single tensor."
        assert exact_conv, msg

        self.lbp_modules = nn.ModuleList([
            _LBPConv2D(kernel_size=ksz, exact_conv=exact_conv,
                       padding_mode=padding_mode, normalize=normalize) for
            ksz in kernel_sizes
        ])

    def forward(self, x):
        """
        Forward function.
        :param x: tensor of shape (batch_size, c, h, w).
        :return: tensor of shape (batch_size, c*nbr_kernels, hp, wp). where
        nbr_kernels are the number of filters (scales)
        used == len(self.kernel_sizes). this assumes that
        """
        msg = "exact_conv must be true to be able to concatenate the lbp " \
              "fetaures in one single tensor."
        assert self.exact_conv, msg

        batch_size, c, h, w = x.shape
        out = None
        for i in range(len(self.kernel_sizes)):
            if i == 0:
                out = self.lbp_modules[i](x)
            else:
                out = torch.cat((out, self.lbp_modules[i](x)), dim=1)

        cd = [batch_size, c * len(self.kernel_sizes), h, w]
        assert list(out.shape) == cd, "unexpected size {}. expected {}.".format(
            list(out.shape), cd
        )

        return out


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed
    seperately for each channel in the input using a depthwise convolution.

    BATCHABLE.
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, exact_conv=True,
                 padding_mode='reflect'):
        """"
        :param channels: (int, sequence): Number of channels of the input
        tensors.  Output will  have this number of channels as well.
        :param kernel_size: (int, sequence): Size of the gaussian kernel.
        :param sigma: (float, sequence): Standard deviation of the gaussian
        kernel.
        :param dim: (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
        :param exact_conv: bool. if True, we pad the input tensor so the
        convlution will output a tensor with the same size as the input.
        :param padding_mode: str. padding mode: 'reflect', 'replicate', or
        'circular'.
        """
        super(GaussianSmoothing, self).__init__()

        allowed_padd_modes = ['reflect', 'replicate', 'circular']
        msg = "'padding_mode'={} not allowed. allowed:{}.".format(
            padding_mode, allowed_padd_modes)
        assert padding_mode in allowed_padd_modes, msg

        self.padding_sz = 0
        self.padding_mode = padding_mode
        self.exact_conv = exact_conv
        if exact_conv:
            self.padding_sz = int((kernel_size - 1) / 2)
            msg = "you asked for exact convolution. for this, the size of " \
                  "the kernel must be odd. found {}.".format(kernel_size)
            assert (kernel_size - 1) % 2 == 0, msg


        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the gaussian function of each
        # dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. '
                'Found {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on. shape:
            batch_size, c, h, w.
        Returns:
            filtered (torch.Tensor): Filtered output. with the same size as
            the input.
        """
        if self.exact_conv:
            return self.conv(
                F.pad(input=input, pad=[self.padding_sz] * 4,
                      mode=self.padding_mode),
                weight=self.weight, groups=self.groups)
        else:
            return self.conv(input, weight=self.weight, groups=self.groups)


class _SobelFilter2D(nn.Module):
    """
    Apply Sobel filter channel wise.
    BATCHABLE.
    """
    def __init__(self, channels, exact_conv=True, padding_mode='reflect'):
        """"
        :param channels: (int, sequence): Number of channels of the input
        tensors.  Output will  have twice number of input channels.
        :param exact_conv: bool. if True, we pad the input tensor so the
        convlution will output a tensor with the same height and width as the
        input.
        :param padding_mode: str. padding mode: 'reflect', 'replicate', or
        'circular'.
        """
        super(_SobelFilter2D, self).__init__()

        allowed_padd_modes = ['reflect', 'replicate', 'circular']
        msg = "'padding_mode'={} not allowed. allowed:{}.".format(
            padding_mode, allowed_padd_modes)
        assert padding_mode in allowed_padd_modes, msg

        self.padding_sz = 0
        self.padding_mode = padding_mode
        self.exact_conv = exact_conv
        kernel_size = [3, 3]
        if exact_conv:
            self.padding_sz = int((kernel_size[0] - 1) / 2)
            msg = "you asked for exact convolution. for this, the size of " \
                  "the kernel must be odd. found {}.".format(kernel_size[0])
            assert (kernel_size[0] - 1) % 2 == 0, msg

        mat = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]], dtype=torch.float, requires_grad=False)

        kernel = torch.cat((
            mat.unsqueeze(0),
            mat.t().unsqueeze(0)
        ), dim=0).unsqueeze(1)
        # shape: [2, 1, 3, 3]

        self.register_buffer('weight', kernel)
        self.conv = F.conv2d

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on. shape:
            batch_size, c, h, w.
        Returns:
            filtered (torch.Tensor): Filtered output. with the same size as
            the input but twice the number of channels c. the height and the
            width of the output are similar to the input's only when
            self.exact_conv is true.
        """
        assert x.ndim == 4, "'x,dim' must be 4. found {}.".format(x.ndim)

        if self.exact_conv:
            padded = F.pad(input=x, pad=[self.padding_sz] * 4,
                           mode=self.padding_mode)
            bs, c, h, w = padded.shape
            filtered = self.conv(padded.view(bs * c, 1, h, w),
                                 weight=self.weight, groups=1)
            bs, c, h, w = x.shape
            return filtered.view(bs, c * 2, h, w)
        else:
            bs, c, h, w = x.shape
            filtered = self.conv(
                x.view(bs * c, h, w), weight=self.weight, groups=1)
            bs, c, h, w = filtered.shape
            return filtered.view(bs, c * 2, h, w)


class _DeepProjectionProp(nn.Module):
    """
    Computes properties based on deep projection.
    """
    def __init__(self, projector):
        """
        Init. function.
        :param projector: instance of nn.Module. a neural model to perform
        the projection of an image and output a vector of a fixed size.
        """
        super(_DeepProjectionProp, self).__init__()
        assert projector is not None, "'projector' is None."
        self.projector = projector

    def forward(self, x):
        """
        Forward function.
        """
        return self.projector(x)


class _HighOrderMomentsProp(nn.Module):
    """
    Computes high order central moments.
    Supported moments: 0, 1, 2. == area, mean, covariance.

    mask: positive (>= 0) of size (h, w).
    Moments:

    Order 0: area of the mask = sum mask(i,j). if normalize_area_msk,
    we normalize this are by the size of the mask: h x w.

    Order 1: mean (scalar: for grey images. vector rgb: for color images.).

    Order 2: covariance matrix of size (3, 3) for rgb color images and the
    variance for grey images.
    """
    def __init__(self, min_order_moment, max_order_moment,
                 normalize_area_msk=True):
        """
        Init. function.
        :param min_order_moment: int. minimum moment.
        :param max_order_moment: int. maximum moment.
        """
        super(_HighOrderMomentsProp, self).__init__()

        msg = "'min_order_moment' must be >=0. found {}.".format(
            min_order_moment)
        assert min_order_moment >= 0, msg

        msg = "'min_order_moment' must be int. found {}.".format(
            type(min_order_moment))
        assert isinstance(min_order_moment, int), msg

        msg = "'max_order_moment' must be >= 'min_order_moment'. " \
              "found max = {}, min = {}.".format(
               max_order_moment, min_order_moment)
        assert max_order_moment >= min_order_moment, msg

        msg = "'max_order_moment' must be int. found {}.".format(
            type(max_order_moment))
        assert isinstance(max_order_moment, int), msg

        msg = "'max_order_moment' must be <=2. found {}.".format(
            max_order_moment)
        assert max_order_moment <= 2, msg

        self.min_order_moment = min_order_moment
        self.max_order_moment = max_order_moment
        self.normalize_area_msk = normalize_area_msk

    def forward(self, x, mask):
        """
        Forward function.
        Compute central moments.
        :param x: input tensor (ALREADY MASKED image) of shape (c, h,
        w) where c is the  number of plans.
        :param mask: tensor of shape (1, h, w). the mask. mask needs to be
        positive (>= 0).
        """
        assert x.ndim == 3, "'xndim' must be 3. found {}.".format(x.ndim)
        assert mask.ndim == 3, "'mask.ndim' must be 3. found {}.".format(
            mask.ndim)

        c, h, w = x.shape

        assert c in [1, 3], "c must be 1 or 3. found {}.".format(c)

        assert mask.shape[1] == h, "mask height {} mismatch x's {}.".format(
            mask.shape[1],  x.shape[1]
        )
        assert mask.shape[2] == w, "mask width {} mismatch x's {}.".format(
            mask.shape[2], x.shape[2]
        )

        nbr = float(mask.sum())
        fact = 1. / float(nbr - 1)  # unbiased estimation.
        size = float(h * w)
        ord0, ord1, ord2 = None, None, None
        out = None
        for i in range(self.min_order_moment, self.max_order_moment + 1):
            if i == 0:
                ord0 = mask.sum().view(1)
                if self.normalize_area_msk:
                    ord0 = (1. * ord0) / size

                out = ord0.view(1)
            elif i == 1:
                if c == 1:
                    ord1 = x.view(-1).sum().view(-1) * fact
                elif c == 3:
                    ord1 = x.view(c, -1).sum(dim=1).view(c) * fact

                if out is None:
                    out = ord1
                else:
                    out = torch.cat((out, ord1))
            elif i == 2:
                if ord1 is None:
                    if c == 1:
                        avg = x.view(-1).sum().view(-1) * fact
                    elif c == 3:
                        avg = x.view(c, -1).sum(dim=1).view(c) * fact
                else:
                    avg = ord1
                ord2 = x.view(c, -1) - avg.view(-1, 1)  # central moment

                ord2 = ord2.matmul(ord2.t()).view(-1) * fact  # covariance
                # matrix of shape (c*c).view(-1) = c*c.
                if out is None:
                    out = ord2
                else:
                    out = torch.cat((out, ord2))
            else:
                raise ValueError(
                    '{}: Unsupported central order higher than 2.'.format(i))

        return out


class SoftHistogram(nn.Module):
    """
    Computes a (soft)-histogram of `torch.histc`.
    Histograms are not differentiable.
    We use an approximation based on binning.

    NOTE:
    OPERATES ON BATCHES (EXPECTED THE FIRST DIM TO BE THE BATCH SIZE).

    Ref:
    https://discuss.pytorch.org/t/differentiable-torch-histc/25865/9
    """
    def __init__(self, bins=256, min=0., max=1., sigma=1e5):
        """
        Init. function.
        :param bins: int. number of bins.
        :param min: int or float. minimum possible value.
        :param max: int or float. maximum possible value.
        :param sigma: float. the heat of the sigmoid. the higher the value,
        the sharper the sigmoid (~ the better the approximation).
        """
        super(SoftHistogram, self).__init__()

        assert bins != 0, "'bins' can not be 0."

        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = float(sigma)
        self.delta = float(max - min) / float(bins)
        self.register_buffer(
            "centers",
            float(min) + self.delta * (torch.arange(bins).float() + 0.5))

    def forward(self, x, mask=None):
        """
        Forward function.
        Computes histogram.
        :param x: vector, pytorch tensor. of shape (batch_size, n).
        :param mask: vector, pytorch tensor or None. same size as x. can be
        used to exclude some components from x. ideally, x should be in {0,
        1}. continuous values in [0, 1] are accepted.
        :return: pytorch tensor of shape (batch_size, bins).
        """
        assert x.ndim == 2, "'x.ndim' must be 2. found {}.".format(x.ndim)

        x_shape = x.shape
        x = x.unsqueeze(1) - self.centers.unsqueeze(1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2.)) - torch.sigmoid(
            self.sigma * (x - self.delta/2.))
        # x shape: batch_size, bins, number of components in x.
        if mask is not None:
            msg = "shape mismatch x={}, mask={}.".format(x.shape, mask.shape)
            assert x_shape == mask.shape, msg
            # unsqueeze for the batch size.
            x = x * mask.unsqueeze(1)

        x = x.sum(dim=-1)
        return x


class _HistogramProp(nn.Module):
    """
    Computes a (soft)-histogram of any tensor of shape (c, h, w) where c is
    the number of plans; and a mask of shape (1, h, w).
    NOTE:
        NOTE BATCHABLE (on purpose).
    """
    def __init__(self, bins=256, min=0., max=1., sigma=1e5, normalize=True):
        """
        Init. function.
        :param bins: int. number of bins.
        :param min: float. lower bound value of the histogram.
        :param max: float. the upper bound value of the histogram.
        :param sigma: float, > 0. sigma for SoftHistogram.
        :param normalize: bool. if true, we normalize the histogram to be a
        probability distribution.
        """
        super(_HistogramProp, self).__init__()

        self.histc = SoftHistogram(bins=bins, min=min, max=max, sigma=sigma)
        self.epsilon = 1e-8
        self.normalize = normalize

    def forward(self, x, mask):
        """
        Forward function.
        :param x: input tensor (ALREADY MASKED image) of shape (c, h,
        w) where c is the  number of plans.
        :param mask: tensor of shape (1, h, w). the mask. mask needs to be
        positive (>= 0).
        :return: tensor of NORMALIZED histograms (probabilities) of size
        (c, bins) that sums to 1 for dim=1.
        """
        assert x.ndim == 3, "'xndim' must be 3. found {}.".format(x.ndim)
        assert mask.ndim == 3, "'mask.ndim' must be 3. found {}.".format(
            mask.ndim)

        c, h, w = x.shape

        assert mask.shape[0] == 1, "mask.shape[0] must be 1. " \
                                   "found {}.".format(mask.shape[0])

        assert mask.shape[1] == h, "mask height {} mismatch x's {}.".format(
            mask.shape[1],  x.shape[1]
        )
        assert mask.shape[2] == w, "mask width {} mismatch x's {}.".format(
            mask.shape[2], x.shape[2]
        )

        # raw image
        hist = self.histc(x=x.view(c, -1), mask=mask.view(1, -1).repeat(c, 1))

        # hist normalization.
        if self.normalize:
            hist = hist + self.epsilon
            hist = hist / hist.sum(dim=-1).unsqueeze(1)

        return hist


class _HistogramOfGradientMagnitudesProp(nn.Module):
    """
    Implements:
    `Histogram of gradient magnitudes: A rotation invariant texture-descriptor`,
    https://ieeexplore.ieee.org/document/7351681.

    Computes a (soft)-histogram of any tensor of shape (c, h, w) where c is
    the number of plans; and a mask of shape (1, h, w).

    The gradient is computed using Sobel filter.
    Assumption: the input image is bounded in [0, 1].

    NOTE:
        NOTE BATCHABLE (on purpose).
    """
    def __init__(self, c, bins=256, sigma=1e5, normalize=True,
                 convert_to_grey=False):
        """
        Init. function.
        :param c: int. number of input channels.
        :param bins: int. number of bins.
        :param sigma: float, > 0. sigma for SoftHistogram.
        :param normalize: bool. if true, we normalize the histogram to be a
        probability distribution.
        :param convert_to_grey: bool. if true, the number of channels of the
        input are averaged.
        """
        super(_HistogramOfGradientMagnitudesProp, self).__init__()

        self.min = 0.  # min magnitude.
        self.max = 2. * np.sqrt(2.)  # max magnitude for the gradient of an
        # image computed using Sobel filter under the
        # assumption that the input image is bounded in [0, 1]

        self.histc = SoftHistogram(bins=bins, min=self.min, max=self.max,
                                   sigma=sigma)
        self.epsilon = 1e-8
        self.normalize = normalize
        self.convert_to_grey = convert_to_grey
        self.c = 1 if self.convert_to_grey else c
        self.sobel = _SobelFilter2D(channels=self.c, exact_conv=True,
                                    padding_mode='reflect')

    def forward(self, x, mask):
        """
        Forward function.
        :param x: input tensor (unmasked image) of shape (c, h,
        w) where c is the  number of plans.
        :param mask: tensor of shape (1, h, w). the mask. mask needs to be
        positive (>= 0).
        :return: tensor of NORMALIZED (if self.normalize is true) histograms (
        probabilities) of size (c, bins) that sums to 1 for dim=1.
        """
        assert x.ndim == 3, "'xndim' must be 3. found {}.".format(x.ndim)
        assert mask.ndim == 3, "'mask.ndim' must be 3. found {}.".format(
            mask.ndim)

        c, h, w = x.shape

        assert mask.shape[0] == 1, "mask.shape[0] must be 1. " \
                                   "found {}.".format(mask.shape[0])

        assert mask.shape[1] == h, "mask height {} mismatch x's {}.".format(
            mask.shape[1],  x.shape[1]
        )
        assert mask.shape[2] == w, "mask width {} mismatch x's {}.".format(
            mask.shape[2], x.shape[2]
        )

        min_x = x.min()
        max_x = x.max()
        assert min_x >= 0., "minx expected be >= 0. found {}.".format(min_x)
        assert max_x >= 0., "maxx expected be <= 1. found {}.".format(max_x)

        if self.convert_to_grey:
            x = x.mean(dim=0).unsqueeze(0)  # shape: 1, h, w

        conv_sobel = self.sobel(x.unsqueeze(0))  # shape: 1, sel.c * 2, h, w.
        conv_sobel = conv_sobel.view(self.c, 2, h, w)
        grad = torch.sqrt((conv_sobel ** 2).sum(dim=1)) / self.max  # self.c,
        # h, w
        max_grad = torch.max(grad)
        min_grad = torch.min(grad)
        assert max_grad <= self.max, "max_grad = {}. expected to be <= {}. " \
                                     "not the case.".format(max_grad, self.max)
        assert self.min <= min_grad, "min_grad = {}. expected to be >= {}. " \
                                     "not the case.".format(min_grad, self.min)

        hist = self.histc(
            x=grad.view(self.c, -1), mask=mask.view(1, -1).repeat(self.c, 1)
        )

        # hist normalization.
        if self.normalize:
            hist = hist + self.epsilon
            hist = hist / hist.sum(dim=-1).unsqueeze(1)

        return hist


class Mask(nn.Module):
    """
    Holds the mask as a parameter to be optimized.
    This mask works always with the same image.
    For another image (with different dims (nrr_plans, h, w)), you need to
    instantiate a new mask.
    """
    def __init__(self, c, h, w, smooth=True, kernel_size=17, sigma=10.):
        """
        Init. function.
        :param c: int. number of channels.
        :param h: int. height of the mask.
        :param w: int. width of the mask.
        :param smooth: bool. if true, the mask is smoothed using a gaussian
        kernel.
        :param kernel_size: int. size of the smoothing kernel.
        :param sigma: float. the sigma of the gaussian kernel.
        """
        super(Mask, self).__init__()

        assert c > 0, " c must be > 0. found {}.".format(c)

        self.h = h
        self.w = w
        self.smooth = True
        if self.smooth:
            self.weight = Parameter(torch.ones((1, h, w)) * 0.51)
        else:
            self.weight = Parameter(torch.ones((1, h, w)) * 0.5)
        self.shape_ = [c, h, w]  # the shape of the expected image.
        self.gaussian_smoother = GaussianSmoothing(
            channels=1,
            kernel_size=kernel_size,
            sigma=sigma, dim=2, exact_conv=True,
            padding_mode='reflect')

    def msize(self):
        """
        Return the size of the mask (h, w).

        :return: (h, w) the height and the width of the mask.
        """
        return self.h, self.w

    def get_nb_params(self):
        """
        Count the number of parameters within the model.
        :return: int, number of learnable parameters.
        """
        return sum([p.numel() for p in self.parameters()])

    def check_x(self, x):
        """
        Check that x is ok.
        x, are you ok?
        """
        if x.ndim != 3:
            raise ValueError("x.ndim must be 3. found {}.".format(x.ndim))

        msg = "x shape {} mismaches the expected shape {}.".format(
            x.shape, self.shape_)
        assert list(x.shape) == self.shape_, msg

    def forward(self, x):
        """
        Forward function.
        Mask the input image.
        :param x: tensor. input image. can be gray (2d), or with colors (3
        plans). If x.shape[0] = 3: format is (nb_plans, h, w) for x.
        if x.shape[0] = 1,  x format is (1, h, w).
        """
        self.check_x(x)
        # we have to reshape here...
        # https://discuss.pytorch.org/t/repeat-a-nn-parameter-for-efficient-
        # computation/25659/8
        # repeat seems to cost almost nothing.
        c = x.shape[0]
        if self.smooth:
            return torch.clamp(
                self._get_smooth_weight().repeat(c, 1, 1),
                0., 1.) * x
        else:
            return torch.clamp(self.weight.repeat(c, 1, 1), 0., 1.) * x

    def _get_smooth_weight(self):
        """
        Return the smooth version of the mask.
        :return:
        """
        assert self.smooth, "something is wrong."
        return self.gaussian_smoother(self.weight.unsqueeze(0)).squeeze(0)

    def get_binary_mask(self):
        """
        Returns the binary mask of size (1, h, w).
        """
        if self.smooth:
            return self.binarize_mask(
                torch.clamp(self._get_smooth_weight(), 0., 1.))
        else:
            return self.binarize_mask(torch.clamp(self.weight, 0., 1.))

    def binarize_mask(self, mask):
        """
        Binarize a mask.
        :param mask: pytorch tensor.
        :return: binarize mask of thr same shape as input mask.
        """
        return ((mask >= 0.5) * 1.).float()

    def get_mask(self):
        """
        Return the mask.
        """
        if self.smooth:
            return torch.clamp(self._get_smooth_weight(), 0., 1.)
        else:
            return torch.clamp(self.weight, 0., 1.)

    def get_resized_mask(self, h, w):
        """
        Get a resized version of the mask with size (1, h, w).
        :param h: int. the new height.
        :param w: int. the new width.
        :return:
        """
        msg = "h an w must be both int. h={}, w={}.".format(h, w)
        assert all([h is not None, w is not None]), msg

        if self.msize() != [h, w]:
            return F.interpolate(
                self.get_mask().unsqueeze(0), (h, w), mode='bilinear',
                align_corners=True).squeeze(0)
        else:
            return self.get_mask()

    def save_continuous_mask(self, path, h=None, w=None):
        """
        Store the continuous mask on disc.
        :param path: absolute path. path where to store the mask.
        :param h: int or None. the height to which to resize the mask to
        before storing. if None, w must be None as well.
        :param w: int or none. the width to which to resize the mask before
        storing it. if none, h must be none as well.
        :return:
        """
        msg = "h an w must be both either int or none. h={}, w={}.".format(h, w)
        assert all([h is None, w is None]) or all([h is not None, w is not
                                                   None]), msg
        if [h, w] != [None, None]:
            new_mask = self.get_resized_mask(h, w)
        else:
            new_mask = self.get_mask()

        torch.save(new_mask.detach().cpu(), path)

    def save_binary_mask_image(self, path, h=None, w=None):
        """
        Store the binary mask as an image on disc.
        :param path: absolute path. path where to store the mask.
        :param h: int or None. the height to which to resize the mask to
        before storing. if None, w must be None as well.
        :param w: int or none. the width to which to resize the mask before
        storing it. if none, h must be none as well.
        :return:
        """
        msg = "h and w must be both either int or none. h={}, w={}.".format(
            h, w)
        assert all([h is None, w is None]) or all([h is not None, w is not
                                                   None]), msg
        if [h, w] != [None, None]:
            new_mask = self.get_resized_mask(h, w)
        else:
            new_mask = self.get_mask()

        bin_mask = self.binarize_mask(new_mask)
        bin_mask = bin_mask.detach().cpu().squeeze().numpy()
        # issue with mode=1...
        # https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-
        # from-array-with-mode-1
        img_mask = Image.fromarray(bin_mask.astype(np.uint8) * 255,
                                   mode='L').convert('1')
        img_mask.save(path)

    def get_binary_resized_mask(self, h=None, w=None):
        """
        Return a pytorch binary mask that has been resized if the resize is
        requested.

        Note: you need to detach the mask yourself.

        :param h: int or None. the new height.
        :param w: int or None. the new width.
        :return: pytorch tensor of shape (1, h, 1).
        """
        if (w is not None) and (h is not None):
            new_mask = self.get_resized_mask(h, w)
        else:
            new_mask = self.get_mask()

        return self.binarize_mask(new_mask)


class Properties(nn.Module):
    """
    Computes statistical properties over masked image regions.
    All the stats. prop. are differentiable.
    """
    def __init__(self, c, deep_prj_prop, histogram_prop,
                 high_order_c_moments,
                 max_order_moment, min_order_moment, normalize_area_msk,
                 multi_down_resolution, down_sample_factor,
                 level_multi_resolution, projector, bins, min, max, sigma,
                 use_lbp_hist, lbp_kernels, hgm, convert_to_grey_hgm
                 ):
        """
        Init. function.
        :param c: int. number of input channels.
        :param deep_prj_prop:
        :param histogram_prop: bool. if true, computes global image histogram.
        :param high_order_c_moments:
        :param max_order_moment:
        :param min_order_moment:
        :param normalize_area_msk:
        :param multi_down_resolution:
        :param down_sample_factor:
        :param level_multi_resolution:
        :param projector:
        :param bins:
        :param min:
        :param max:
        :param sigma:
        """
        super(Properties, self).__init__()

        self.deep_projection_prop = None
        self.high_order_moments_rop = None
        self.histogram = None
        self.lbp = None
        self.c = c
        self.hgm = None

        if deep_prj_prop:
            msg = "you asked for deep projection but 'projector' is None."
            assert projector is not None, msg
            self.deep_projection_prop = _DeepProjectionProp(projector=projector)

        if histogram_prop or use_lbp_hist:
            self.histogram = _HistogramProp(
                bins=bins, min=min, max=max, sigma=sigma,
                normalize=NORMALIZE_HISTO)

        if high_order_c_moments:
            self.high_order_moments_rop = _HighOrderMomentsProp(
                min_order_moment=min_order_moment,
                max_order_moment=max_order_moment,
                normalize_area_msk=normalize_area_msk
            )

        if use_lbp_hist:
            self.lbp = _LBPModule(kernel_sizes=lbp_kernels, exact_conv=True,
                                  padding_mode='reflect', normalize=True)

        if hgm:
            self.hgm = _HistogramOfGradientMagnitudesProp(
                c=c, bins=bins, sigma=sigma, convert_to_grey=convert_to_grey_hgm
            )

    def forward(self, x, masked_x, mask):
        """
        Forward function.
        Forward only one sample.
        Compute different statistical properties over the already masked image.
        :param x: unmasked x. pytorch tensor of the image shaped (c, h, w).
        :param masked_x: pytorch tensor. input image. already masked.
         shape (c, h, w).
        :param mask: pytorch tensor of shape (1, h, w). the mask.
        :return:
            prop_his: tensor of shape (c, bins). each row is a histogram.
        """
        assert masked_x.ndim == 3, "'masked_x.ndim' must be 3, " \
                                   "found {}.".format(masked_x.ndim)

        assert x.ndim == 3, "'x.ndim' must be 3, found {}.".format(x.ndim)

        assert mask.ndim == 3, "'mask.ndim' must be 3. found {}.".format(
            mask.ndim)
        assert mask.shape[0] == 1, "mask.shape[0] must be 1. " \
                                   "found {}.".format(mask.shape[0])

        prop_dprj, prop_hom, prop_his = None, None, None
        prop_lbps, props_hgm = None, None

        if self.deep_projection_prop is not None:
            prop_dprj = self.deep_projection_prop(masked_x.unsqueeze(
                0)).squeeze()

        if self.histogram is not None:
            prop_his = self.histogram(x=x, mask=mask)

        if self.high_order_moments_rop is not None:
            prop_hom = self.high_order_moments_rop(x=masked_x, mask=mask)

        if self.lbp is not None:
            prop_lbps = self.lbp(x.unsqueeze(0)).squeeze(0)  # unmasked lbp.
            # (c`, h, w)
            # cp, h, w = prop_lbps.shape
            # prop_lbps = prop_lbps * mask.repeat(cp, 1, 1)  # masked lbp.
            prop_lbps = self.histogram(x=prop_lbps, mask=mask)

        if self.hgm:
            props_hgm = self.hgm(x=x, mask=mask)

        return prop_dprj, prop_hom, prop_his, prop_lbps, props_hgm


def mask_image(img, mask):
    """
    Mask an image. Operates on pytorch tensors.
    """
    assert img.ndim == 3, "'img.ndim' must be 3, found {}.".format(img.ndim)
    assert mask.ndim == 3, "'mask.ndim' must be 3, found {}.".format(mask.ndim)

    assert mask.shape[0] == 1, "'mask.shape[0]' must be 1 found {}.".format(
        mask.shape[0]
    )

    c = img.shape[0]

    return mask.repeat(c, 1, 1) * img


def test_Mask():
    """
    Test function: Mask().
    """
    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    c, h, w = 3, 20, 30
    m = Mask(c, h, w)
    m.to(DEVICE)
    m.train()
    print("Number of parameters {}".format(m.get_nb_params()))
    print("Number of pixels {}".format(m.weight.numel()))
    print("Weight {}".format(m.weight))
    print("Sum weight {}".format(m.weight.sum()))

    x = torch.rand((c, h, w)).to(DEVICE)
    output = m(x)
    print(output.shape, output.dtype, output.device)
    # saving
    root = '../data/debug/visualization/'
    m.save_continuous_mask(join(root, 'conti.pt'), h=None, w=None)
    m.save_continuous_mask(join(root, 'conti-resized.pt'), h=300, w=200)

    m.save_binary_mask_image(join(root, 'conti.png'), h=None, w=None)
    m.save_binary_mask_image(join(root, 'conti-resized.png'), h=300, w=200)


def test_SoftHistogram():
    """
    Test function: SoftHistogram().
    """
    reset_seed(0)
    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    bins = 256
    min = 0.
    max = 1.
    sigma = 1e5
    m = SoftHistogram(bins=bins, min=min, max=max, sigma=sigma)
    m.to(DEVICE)

    batch_sz = 3
    x = torch.rand((batch_sz, 5)).to(DEVICE)
    print(x[0])

    mask = torch.rand((batch_sz, 5)).to(DEVICE)
    print(mask[0])
    output = m(x, mask=x)
    print(output.shape, output.dtype, output.device)
    print(output.shape)
    better_hist = None
    for i in range(batch_sz):
        if i == 0:
            better_hist = torch.histc(x[i, :], bins=bins, min=min,
                                      max=max).view(1, -1)
        else:
            better_hist = torch.cat(
                (better_hist,
                 torch.histc(x[i, :], bins=bins, min=min, max=max).view(1, -1)),
                dim=0
            )
    print(better_hist.shape)
    print("error: {}".format(torch.abs(output - better_hist).sum(dim=1)))


def test__HistogramProp():
    """
    Test function: _HistogramProp().
    """
    reset_seed(0)
    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    bins = 256
    min = 0.
    max = 1.
    sigma = 1e5
    m = _HistogramProp(bins=bins, min=min, max=max, sigma=sigma)
    m.to(DEVICE)

    c, h, w = 4, 10, 20
    x = torch.rand((c, h, w)).to(DEVICE)

    mask = torch.rand((1, h, w)).to(DEVICE)

    output = m(x, mask=mask)
    print(output.shape, output.dtype, output.device)


def test__HistogramOfGradientMagnitudesProp():
    """
    Test function: _HistogramOfGradientMagnitudesProp().
    """
    reset_seed(0)
    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    bins = 256
    sigma = 1e5
    c, h, w = 4, 10, 20
    m = _HistogramOfGradientMagnitudesProp(c=c, bins=bins, sigma=sigma,
                                           convert_to_grey=False)
    m.to(DEVICE)

    x = torch.rand((c, h, w)).to(DEVICE)
    mask = torch.rand((1, h, w)).to(DEVICE)

    output = m(x, mask=mask)
    print(output.shape, output.dtype, output.device)
    print(output.min(), output.max())

    # test on an image.
    root = '../data/debug/input'
    file = 'Black_Footed_Albatross_0006_796065.jpg'
    path = join(root, file)
    input_img = Image.open(path).convert('RGB')
    totensor = transforms.Compose([transforms.ToTensor()])
    input_img = totensor(input_img).to(DEVICE)
    c, h, w = input_img.shape
    m = _HistogramOfGradientMagnitudesProp(c=c, bins=bins, sigma=sigma,
                                           convert_to_grey=True)
    m.to(DEVICE)
    mask = torch.rand((1, h, w)).to(DEVICE)
    print('test on an image of shape {}.'.format(input_img.shape))
    output = m(input_img, mask=mask)
    print(output.shape, output.dtype, output.device)
    print(output.min(), output.max())
    fig = plt.figure()
    x = list(range(256))
    for i in range(m.c):
        h = output[i].cpu().squeeze().numpy()
        plt.bar(x, h, label="{}".format(i))

    fig.savefig(
        join("../data/debug/visualization/", "hgm-c={}-{}".format(m.c, file)))


def test_GaussianSmoothing():
    """
    Test: GaussianSmoothing()
    :return:
    """
    reset_seed(0, check_cudnn=False)

    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    smoothing = GaussianSmoothing(channels=3, kernel_size=3, sigma=1,
                                  exact_conv=True)
    smoothing.to(DEVICE)

    input = torch.rand(1, 3, 100, 100).to(DEVICE)
    output = smoothing(input)
    print(output.shape, input.shape)
    for p in smoothing.parameters():
        print(p.shape, p.requires_grad)


def test___LBPConv2D():
    """
    Test: _LBPConv2D()
    :return:
    """
    reset_seed(0, check_cudnn=False)

    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    instance = _LBPConv2D(kernel_size=3, exact_conv=True,
                          normalize=True)
    instance.to(DEVICE)
    bs, c, h, w = 1, 1, 20, 30

    x = torch.rand(bs, c, h, w)
    x = x.to(DEVICE)
    out = instance(x)
    for p in instance.parameters():  # expected: no parameters.
        print("p: ", p.shape)
    print(out.shape)
    print(out.min(), out.max())


def test__LBPModule():
    """
    Test: _LBPModule()
    :return:
    """
    reset_seed(0, check_cudnn=False)

    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    instance = _LBPModule(kernel_sizes=[3, 5, 7], exact_conv=True,
                          normalize=True)
    instance.to(DEVICE)
    bs, c, h, w = 1, 5, 20, 30

    x = torch.rand(bs, c, h, w)
    x = x.to(DEVICE)
    out = instance(x)
    for p in instance.parameters():  # expected: no parameters.
        print("p: ", p.shape)
    print(out.shape)
    print(out.min(), out.max())


def test__HighOrderMomentsProp():
    """
    Test function of _HighOrderMomentsProp().
    """
    reset_seed(0, check_cudnn=False)
    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    c, h, w = 3, 20, 30
    x = torch.rand(c, h, w)
    mask = torch.rand(c, h, w)
    x = x.to(DEVICE)
    mask = mask.to(DEVICE)
    instance = _HighOrderMomentsProp(min_order_moment=2, max_order_moment=2,
                                     normalize_area_msk=True)

    t0 = dt.datetime.now()
    output = instance(x=x, mask=mask)
    print("forward time: {}".format(dt.datetime.now() - t0))
    print(output.shape, output.dtype, output.device)
    print(output)


def test__SobelFilter2D():
    """
    Test: _SobelFilter2D()
    :return:
    """
    reset_seed(0, check_cudnn=False)

    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    sobel = _SobelFilter2D(channels=3, exact_conv=True)
    sobel.to(DEVICE)

    input = torch.rand(1, 3, 100, 100).to(DEVICE)
    output = sobel(input)
    print(output.shape, input.shape)
    for p in sobel.parameters():
        print(p.shape, p.requires_grad)


if __name__ == "__main__":
    # test__HistogramOfGradientMagnitudesProp()
    # test__SobelFilter2D()
    # test__LBPModule()
    # test___LBPConv2D()
    # test_GaussianSmoothing()
    test__HistogramProp()
    # test_SoftHistogram()
    # test_Mask()
    # test__HighOrderMomentsProp()
