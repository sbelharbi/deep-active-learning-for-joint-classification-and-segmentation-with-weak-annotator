#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""criteria.py
Implements different learning losses and metrics.
"""
import os
import warnings
import sys
import datetime as dt


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import half_normal as hnormal
from torch.distributions import normal, bernoulli

sys.path.append("..")

import constants
from shared import announce_msg
from shared import check_nans
from reproducibility import reset_seed
import deeplearning.decay as decay


__all__ = ["Entropy", "CE", "KL", "Metrics",
           "Dice", "IOU", "BinSoftInvDiceLoss",
           "BinCrossEntropySegmLoss", "BCEAndSoftDiceLoss", "HybridLoss"]


class _Loss(nn.Module):
    """
    Template loss class.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_Loss, self).__init__()

    def forward(self, scores, labels, targets, masks_pred, masks_trg, tags,
                weights=None, avg=True):
        """
        Forward function.
        :param scores: torch tensor (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :param targets: torch tensor  (n, d). used only for our method for
        classification task.
        :param masks_pred: torch tensor. predicted masks (for seg.).
        normalized scores.
        :param masks_trg: torch tensor. target mask (for seg).
        :param tags: vector of Log integers. The tag of each sample.
        :param weights: vector of weights. a weight per sample.
        :param avg: bool. if true, the loss is averaged otherwise,
        it is summed up.
        :return: real value loss.
        """
        raise NotImplementedError('You need to override this.')

    def __str__(self):
        raise NotImplementedError('You need to override this.')

    def predict_label(self, scores):
        """
        Predict the output label based on the scores or probabilities for
        global classification.

        :param scores: matrix (n, nbr_c) of unormalized-scores or probabilities.
        :return: vector of long integer. The predicted label(s).
        """
        return scores.argmax(dim=1, keepdim=False)

    def predict_pixels(self, masks_pred, threshold=0.5):
        """
        Predict the binary mask for segmentation.

        :param masks_pred: tensor (n, whatever-dims) of normalized-scores.
        :param threshold: float. threshold in [0., 1.].
        :return: tensor of same shape as `masks_pred`. Contains the binary
        mask, thresholded at `threshold`. dtype: float.
        """
        msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
        assert 0 <= threshold <= 1., msg
        msg = "'threshold' type must be float. found {}.".format(
            type(threshold))
        assert isinstance(threshold, float), msg

        return (masks_pred >= threshold).float()


class _CrossEntropy(nn.Module):
    """
    Compute Entropy between two distributions p, q:

    H(p, q) = - sum_i pi * log q_i.

    This is different from torch.nn.CrossEntropy() (and _CE) in the sens that
    _CrossEntropy() operates on probabilities and computes the entire sum
    because the target is not discrete but continuous.

    This loss can be used to compute the entropy of a distribution H(p):

    H(p) = - sum_i pi * log p_i.

    For this purpose use `forward(p, p)`.
    """
    def __init__(self, sumit=True):
        """
        Init. function.
        :param sumit: bool. If True, we sum across the dim=1 to obtain the true
        cross-entropy. In this case, the output is a vector of shape (n) where
        each component is the corresponding cross-entropy.

        If False, we do not sum across dim=1, and return a matrix of shape (
        n, m) where n is the number of samples and m is the number of
        elements in the probability distribution.
        """
        super(_CrossEntropy, self).__init__()

        self.sumit = sumit

    def forward(self, p, q):
        """
        The forward function. It operate on batches.
        :param p: tensor of size (n, m). Each row is a probability
        distribution.
        :param q: tensor of size (n, m). Each row is a probability
        distribution.
        :return: a vector of size (n) or (n, m) dependent on self.sumit.
        """
        if self.sumit:
            return (-p * torch.log(q)).sum(dim=1)
        else:
            return -p * torch.log(q)

    def __str__(self):
        return "{}(): Cross-entropy over continuous target. sumit={}.".format(
            self.__class__.__name__, self.sumit)


class _CE(nn.Module):
    """
    Cross-entropy loss.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_CE, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self,
                scores,
                labels,
                weights
                ):
        """
        Forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :param weights: vector of weights. a weight per sample.
        :return: real value cross-entropy loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        return (self.loss(scores, labels) * weights).mean()

    def __str__(self):
        return "{}(): Cross-entropy over discrete target.".format(
            self.__class__.__name__)


class _MSE(nn.Module):
    """
    Class computes the mean-squarred error between two vectors.
    """
    def __init__(self):
        super(_MSE, self).__init__()

    def forward(self, a, b):
        """
        The forward function.

        Computes MSE.
        :param a: tensor of size (n, m).
        :param b: tensor of size (n, m).
        :return: matrix of differences of size (n, m).
        """
        msg = "Mismatch shapes mse(a, b): a --> {}, b --> {}.".format(
            a.shape, b.shape)
        assert a.shape == b.shape, msg

        assert a.ndim == 2, "Expect 2 dimensions of `a` but found " \
                            "{}.".format(a.ndim)

        return (a - b)**2  # TODO: you may want to sum/mean(dim=1). but for
        # now each component is guaranteed to be >= 0.

    def __str__(self):
        return "{}(): Unormalized mean-squared error.".format(
            self.__class__.__name__)


class _JSD(nn.Module):
    """
    Implements Jensen-Shannon divergence between two distributions.
    """
    def __init__(self):
        super(_JSD, self).__init__()

        self.entropy = _CrossEntropy(sumit=True)

    def forward(self, a, b):
        """
        The forward function.

        Computes JSD(P, Q) = 1/2. * KL(P || (P+Q)/2.) + 1/2. KL(Q || (P+Q)/2.).

        :param a: tensor of size (n, m). Each row is a probability distribution.
        :param b: tensor of size (n, m). Each row is a probability distribution.
        :return: vector of differences of size (n) where each component is
        the Jensen-Shannon divergence.
        """
        msg = "Mismatch shapes JSD(a, b): a --> {}, b --> {}.".format(
            a.shape, b.shape)
        assert a.shape == b.shape, msg
        assert a.ndim == 2, "Expect 2 dimensions of `a` but found " \
                            "{}.".format(a.ndim)

        # average probabilitiy
        avg = (a + b) / 2.
        # KL(p, q) = H(p, q) - H(p).
        # JSD(p, q) = KL(p, (p+q)/2.) / 2. + KL(p, (p+q)/2.) / 2.
        jsd = (self.entropy(a, avg) - self.entropy(a, a)) / 2. + (
                self.entropy(b, avg) - self.entropy(b, b)) / 2.

        # check the that 0 <= jsd <= 1.
        min_ = jsd.min()
        max_ = jsd.max()
        cond = (0. <= min_ <= 1.) and (0. <= max_ <= 1.)
        # this assertion can lead to error because of floating points
        # imprecision. eemple: -2.3842e-07.
        # assert cond, "JSD values are not all in [0, 1]. {}".format(jsd)
        # clip values into [0, 1]

        return torch.clamp(jsd, min=0., max=1.)

    def __str__(self):
        return "{}(): Jensen-Shannon divergence.".format(
            self.__class__.__name__)


class LossHistogramsMatching(nn.Module):
    """
    Loss computed by matching two normalized histograms: predicted histogram and
    reference histogram.
    """
    def __init__(self):
        """
        Init. function
        """
        super(LossHistogramsMatching, self).__init__()

        self.measure = _JSD()

    def forward(self, trg_his, src_his):
        """
        Forward function.
        Computes the loss of matching trg_his into src_his. each is a
        normalized histogram (probabilities).
        :param trg_his: pytorch tensor of size (batch_size, bins) of the
        predicted histogram.
        :param src_his: pytorch tensor of size (batch_size, bins) of the
        reference histogram.
        :returns: torch vector of size (batch_size) where each element is the
        `distance` between the two histograms in the corresponding index.
        """
        assert trg_his.ndim == 2, "'trg_his' must have 2 dims. " \
                                  "found {}.".format(trg_his.ndim)
        assert src_his.ndim == 2, "'src_his' must have 2 dims. " \
                                  "found {}.".format(src_his.ndim)
        assert src_his.shape == trg_his.shape, "shapes mismatch. src={}, " \
                                               "trg={}.".format(src_his.shape,
                                                                trg_his.shape)
        if isinstance(self.measure, nn.KLDivLoss):
            return self.measure(torch.log(trg_his), src_his).sum(dim=1)
        elif isinstance(self.measure, _JSD):
            return self.measure(trg_his, src_his)
        else:
            raise NotImplementedError


class _LossExtendedLB(nn.Module):
    """
    Extended log-barrier loss (ELB).
    Optimize inequality constraint : f(x) <= 0.

    Refs:
    1. Kervadec, H., Dolz, J., Yuan, J., Desrosiers, C., Granger, E., and Ben
     Ayed, I. (2019b). Constrained deep networks:Lagrangian optimization
     via log-barrier extensions.CoRR, abs/1904.04205
    2. S. Belharbi, I. Ben Ayed, L. McCaffrey and E. Granger,
    “Deep Ordinal Classification with Inequality Constraints”, CoRR,
    abs/1911.10720, 2019.
    """
    def __init__(self,
                 init_t=1.,
                 max_t=10.,
                 mulcoef=1.01
                 ):
        """
        Init. function.

        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        """
        super(_LossExtendedLB, self).__init__()

        msg = "`mulcoef` must be a float. You provided {} ....[NOT OK]".format(
            type(mulcoef))
        assert isinstance(mulcoef, float), msg
        msg = "`mulcoef` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(mulcoef)
        assert mulcoef > 0., msg

        msg = "`init_t` must be a float. You provided {} ....[NOT OK]".format(
            type(init_t))
        assert isinstance(init_t, float), msg
        msg = "`init_t` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(init_t)
        assert init_t > 0., msg

        msg = "`max_t` must be a float. You provided {} ....[NOT OK]".format(
            type(max_t))
        assert isinstance(max_t, float), msg
        msg = "`max_t` must be > `init_t`. float. You provided {} " \
              "....[NOT OK]".format(max_t)
        assert max_t > init_t, msg

        self.init_t = init_t

        self.register_buffer(
            "mulcoef", torch.tensor([mulcoef], requires_grad=False).float())
        # create `t`.
        self.register_buffer(
            "t_lb", torch.tensor([init_t], requires_grad=False).float())

        self.register_buffer(
            "max_t", torch.tensor([max_t], requires_grad=False).float())

    def set_t(self, val):
        """
        Set the value of `t`, the hyper-parameter of the log-barrier method.
        :param val: float > 0. new value of `t`.
        :return:
        """
        msg = "`t` must be a float. You provided {} ....[NOT OK]".format(
            type(val))
        assert isinstance(val, float) or (isinstance(val, torch.Tensor) and
                                          val.ndim == 1 and
                                          val.dtype == torch.float), msg
        msg = "`t` must be > 0. float. You provided {} ....[NOT OK]".format(val)
        assert val > 0., msg

        if isinstance(val, float):
            self.register_buffer(
                "t_lb", torch.tensor([val], requires_grad=False).float()).to(
                self.t_lb.device
            )
        elif isinstance(val, torch.Tensor):
            self.register_buffer("t_lb", val.float().requires_grad_(False))

    def get_t(self):
        """
        Returns the value of 't_lb'.
        """
        return self.t_lb

    def update_t(self):
        """
        Update the value of `t`.
        :return:
        """
        self.set_t(torch.min(self.t_lb * self.mulcoef, self.max_t))

    def forward(self, fx):
        """
        The forward function.
        :param fx: pytorch tensor. a vector.
        :return: real value extended-log-barrier-based loss.
        """
        assert fx.ndim == 1, "fx.ndim must be 1. found {}.".format(fx.ndim)

        loss_fx = fx * 0.

        # vals <= -1/(t**2).
        ct = - (1. / (self.t_lb**2))

        idx_less = ((fx < ct) | (fx == ct)).nonzero().squeeze()
        if idx_less.numel() > 0:
            val_less = fx[idx_less]
            loss_less = - (1. / self.t_lb) * torch.log(- val_less)
            loss_fx[idx_less] = loss_less

        # vals > -1/(t**2).
        idx_great = (fx > ct).nonzero().squeeze()
        if idx_great.numel() > 0:
            val_great = fx[idx_great]
            loss_great = self.t_lb * val_great - (1. / self.t_lb) * \
                torch.log((1. / (self.t_lb**2))) + (1. / self.t_lb)
            loss_fx[idx_great] = loss_great

        loss = loss_fx.sum()

        return loss

    def __str__(self):
        return "{}(): extended-log-barrier-based method.".format(
            self.__class__.__name__)



# ==============================================================================
#                          PUBLIC LOSSES
#                 1. LossCE: Cross-entropy loss.
# ==============================================================================


class Entropy(_CrossEntropy):
    """
    Class that computes the entropy of a distribution p:
    entropy = - sum_i p_i * log(p_i).
    """
    def __init__(self):
        """
        Init. function.
        """
        super(Entropy, self).__init__(sumit=True)

    def forward(self, p):
        """
        Forward function.
        :param p: tensor of probability distribution (each row).
        """
        return super(Entropy, self).forward(p, p)


class CE(_Loss):
    """
    Cross-entropy loss for CL and SEG tasks. Requires full supervision of the
    global labels.
    Used for other methods.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(CE, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self,
                scores,
                labels,
                targets=None,
                masks_pred=None,
                masks_trg=None,
                tags=None,
                weights=None,
                avg=True
                ):
        """
        Forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :param targets: torch tensor  (n, d). used only for our method. must
        be None for this class.
        :param masks_pred: torch tensor. predicted masks (for seg.). must be
        None for this class.
        :param masks_trg: torch tensor. target mask (for seg). must be None
        for this class.
        :param tags: vector of Log integers. The tag of each sample.
        :param weights: vector of weights. a weight per sample.
        :param avg: bool. if true, the loss is averaged otherwise,
        just summed up.
        :return: real value cross-entropy loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"
        assert targets is None, "'targets' must be None"
        assert masks_pred is None, "'masks_pred' must be None"
        assert masks_trg is None, "'masks_trg' must be None"

        if weights is not None:
            return (self.loss(scores, labels) * weights).mean()
        else:
            if avg:
                return (self.loss(scores, labels)).mean()
            else:
                return (self.loss(scores, labels)).sum()

    def __str__(self):
        return "{}(): Cross-entropy over discrete target.".format(
            self.__class__.__name__)


class KL(_Loss):
    """
    KL-divergence for CL task only (for our method only).
    """
    def __init__(self):
        """
        Init. function.
        """
        super(KL, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self,
                scores,
                labels,
                targets,
                tags,
                masks_pred=None,
                masks_trg=None,
                weights=None,
                avg=True
                ):
        """
        Forward function.
        targets it not supposed to be None.
        if it is None, this means that we are evaluating the loss over a
        validation set. In this case, we use standard cross-entropy.
        """
        if targets is None:
            if avg:
                return self.ce(scores, labels).mean()
            else:
                return self.ce(scores, labels).sum()

        # if not:
        # compute the log_softmax to prepare the input of kl.
        logsfmx = F.log_softmax(scores, dim=1)
        if avg:
            return self.kl(logsfmx, targets).sum(dim=1).mean()
        else:
            return self.kl(logsfmx, targets).sum(dim=1)

    def __str__(self):
        return "{}(): KL-div for CL task (Ours).".format(
            self.__class__.__name__)


class Dice(nn.Module):
    """
    Computes Dice index for binary classes.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(Dice, self).__init__()

        self.smooth = 1e-8  # prevent dividing by zero. some samples are
        # mistakenly labeled by a black mask (no target). in this case,
        # we obtain a dice of 0. We can do nothing about this. moving on.

    def forward(self, pred_m, true_m):
        """
        Forward function.
        Computes Dice index [0, 1] for binary classes.
        :param pred_m: predicted mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :return vector of size (n) contains Dice index of each sample. values
        are in [0, 1].
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        pflat = pred_m
        tflat = true_m
        intersection = (pflat * tflat).sum(dim=1)

        return (2. * intersection) / (pflat.sum(dim=1) + tflat.sum(dim=1) +
                                      self.smooth)

    def __str__(self):
        return "{}(): Dice index.".format(self.__class__.__name__)


class IOU(nn.Module):
    """
    Computes the IOU (intersection over union) metric for one class.
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        :param smooth: float > 0. smoothing value.
        """
        super(IOU, self).__init__()

        assert smooth > 0., "'smooth' must be > 0. found {}.".format(smooth)
        msg = "'smooth' type must be float, found {}.".format(type(smooth))
        assert isinstance(smooth, float), msg
        self.smooth = smooth

    def forward(self, pred_m, true_m):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :return vector of size (n) contains IOU metric of each sample.
        values are in [0, 1] where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        pflat = pred_m
        tflat = true_m
        intersection = (pflat * tflat).sum(dim=1)
        union = pflat.sum(dim=1) + tflat.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou

    def __str__(self):
        return "{}(): IOU metric for one class. " \
               "".format(self.__class__.__name__)



class _SegLoss(nn.Module):
    """
    Base class for a segmentation loss.
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        :param smooth:
        """
        super(_SegLoss, self).__init__()


class BinSoftInvDiceLoss(_SegLoss):
    """
    Computes the soft inverse Dice index loss for binary segmentation.
    Ref: https://arxiv.org/pdf/1606.04797.pdf
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        :param smooth: float > 0. smoothing value.
        """
        super(BinSoftInvDiceLoss, self).__init__()

        assert smooth > 0., "'smooth' must be > 0. found {}.".format(smooth)
        msg = "'smooth' type must be float, found {}.".format(type(smooth))
        assert isinstance(smooth, float), msg
        self.smooth = smooth

    def forward(self,
                pred_m,
                true_m,
                gate=None
                ):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param gate: (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels) OR None.
        if not none, each value indicated if to ignore the pixel-annotation (
        value=0) or not (value=1). This is an abstention gate.
        :return vector of size (n) contains inverse Dice index (i.e.,
        1 - dice) of each sample. values are in [0, 1] where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        if gate is not None:
            assert gate.ndim == 2, "'gate.ndim' = {}. must be {}.".format(
                gate.ndim, 2)

            msg = "size mismatches: {}, {}.".format(true_m.shape, pred_m.shape)
            assert true_m.shape == gate.shape, msg

            msg = "'gate' dtype required is torch.float. found {}.".format(
                gate.dtype)
            assert gate.dtype == torch.float, msg


        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg

        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg


        pflat = pred_m if gate is None else pred_m * gate
        tflat = true_m if gate is None else true_m * gate
        intersection = (pflat * tflat).sum(dim=1)

        dice = (2. * intersection + self.smooth) / (
                (pflat**2).sum(dim=1) + (tflat**2).sum(dim=1) + self.smooth)

        return 1. - dice

    def __str__(self):
        return "{}(): Soft binary inverse Dice index loss for binary " \
               "segmentation.".format(self.__class__.__name__)


class BinCrossEntropySegmLoss(_SegLoss):
    """
    Implement the binary cross-entropy for segmentation.
    https://pytorch.org/docs/1.4.0/nn.html#torch.nn.BCELoss
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        """
        super(BinCrossEntropySegmLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self,
                pred_m,
                true_m,
                gate=None
                ):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param gate: (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels) OR None.
        if not none, each value indicated if to ignore the pixel-annotation (
        value=0) or not (value=1). This is an abstention gate.
        :return vector of size (n) contains the binary cross entropy. values
        are in [0, +inf[ where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        if gate is not None:
            assert gate.ndim == 2, "'gate.ndim' = {}. must be {}.".format(
                gate.ndim, 2)

            msg = "size mismatches: {}, {}.".format(true_m.shape, pred_m.shape)
            assert true_m.shape == gate.shape, msg

            msg = "'gate' dtype required is torch.float. found {}.".format(
                gate.dtype)
            assert gate.dtype == torch.float, msg

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        if gate is None:
            return self.bce(input=pred_m, target=true_m).mean(dim=1)
        else:
            tmp = self.bce(input=pred_m, target=true_m) * gate
            # average only over the non-ignored points.
            tmp = tmp.sum(dim=1) / gate.sum(dim=1)
            return tmp

    def __str__(self):
        return "{}(): Binary cross-entropy loss for binary " \
               "segmentation.".format(self.__class__.__name__)


class BinL1SegmLoss(_SegLoss):
    """
    Implement the binary L1 for segmentation.
    Ghosh, Aritra, Himanshu Kumar, and P. S. Sastry.
    "Robust loss functions under label noise for deep neural networks."
    Thirty-First AAAI Conference on Artificial Intelligence. 2017.
    """
    def __init__(self,smooth=1.):
        """
        Init. function.
        """
        super(BinL1SegmLoss, self).__init__()

    def forward(self,
                pred_m,
                true_m,
                gate=None
                ):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param gate: (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels) OR None.
        if not none, each value indicated if to ignore the pixel-annotation (
        value=0) or not (value=1). This is an abstention gate.
        :return vector of size (n) contains the average mean absolute error.
        values  are in [0, 1] where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        if gate is not None:
            assert gate.ndim == 2, "'gate.ndim' = {}. must be {}.".format(
                gate.ndim, 2)

            msg = "size mismatches: {}, {}.".format(true_m.shape, pred_m.shape)
            assert true_m.shape == gate.shape, msg

            msg = "'gate' dtype required is torch.float. found {}.".format(
                gate.dtype)
            assert gate.dtype == torch.float, msg

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        if gate is None:
            return torch.abs(pred_m - true_m).mean(dim=1)
        else:
            tmp = torch.abs(pred_m - true_m) * gate
            # average only over the non-ignored points.
            tmp = tmp.sum(dim=1) / gate.sum(dim=1)
            return tmp

    def __str__(self):
        return "{}(): Binary L1 loss for binary " \
               "segmentation.".format(self.__class__.__name__)


class BinIOUSegmLoss(_SegLoss):
    """
    Computes the IOU loss for binary segmentation.
    Robust to noisy labels.
    Rister, Blaine, et al. "CT organ segmentation using GPU data augmentation,
    unsupervised labels and IOU loss." arXiv preprint arXiv:1811.11226 (2018).
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        :param smooth: float > 0. smoothing value.
        """
        super(BinIOUSegmLoss, self).__init__()

        assert smooth > 0., "'smooth' must be > 0. found {}.".format(smooth)
        msg = "'smooth' type must be float, found {}.".format(type(smooth))
        assert isinstance(smooth, float), msg
        self.smooth = smooth

    def forward(self,
                pred_m,
                true_m,
                gate=None
                ):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param gate: (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels) OR None.
        if not none, each value indicated if to ignore the pixel-annotation (
        value=0) or not (value=1). This is an abstention gate.
        :return vector of size (n) contains inverse Dice index (i.e.,
        1 - dice) of each sample. values are in [0, 1] where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        if gate is not None:
            assert gate.ndim == 2, "'gate.ndim' = {}. must be {}.".format(
                gate.ndim, 2)

            msg = "size mismatches: {}, {}.".format(true_m.shape, pred_m.shape)
            assert true_m.shape == gate.shape, msg

            msg = "'gate' dtype required is torch.float. found {}.".format(
                gate.dtype)
            assert gate.dtype == torch.float, msg

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        pflat = pred_m if gate is None else pred_m * gate
        tflat = true_m if gate is None else true_m * gate
        intersection = (pflat * tflat).sum(dim=1)
        union = pflat.sum(dim=1) + tflat.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1. - iou

    def __str__(self):
        return "{}(): Soft IUO loss for binary " \
               "segmentation.".format(self.__class__.__name__)


class BinL1SegAndIOUSegmLoss(_SegLoss):
    """
    Implement the segmentation loss that is the sum of the binary
    IOU and L1 segmentation loss.
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        """
        super(BinL1SegAndIOUSegmLoss, self).__init__()
        self.local_loss = BinL1SegmLoss()
        self.global_loss = BinIOUSegmLoss(smooth=smooth)

    def forward(self,
                pred_m,
                true_m,
                gate=None
                ):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param gate: (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels) OR None.
        if not none, each value indicated if to ignore the pixel-annotation (
        value=0) or not (value=1). This is an abstention gate.
        :return vector of size (n) contains the binary cross entropy and
        soft inverse Dice loss. values are in [0, +inf[ where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        if gate is not None:
            assert gate.ndim == 2, "'gate.ndim' = {}. must be {}.".format(
                gate.ndim, 2)

            msg = "size mismatches: {}, {}.".format(true_m.shape, pred_m.shape)
            assert true_m.shape == gate.shape, msg

            msg = "'gate' dtype required is torch.float. found {}.".format(
                gate.dtype)
            assert gate.dtype == torch.float, msg

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        return self.local_loss(
            pred_m=pred_m, true_m=true_m, gate=gate) + self.global_loss(
            pred_m=pred_m, true_m=true_m, gate=gate)

    def __str__(self):
        return "{}(): Binary L1 Seg. loss  + soft IOU " \
               "loss for binary segmentation.".format(self.__class__.__name__)


class BCEAndSoftDiceLoss(_SegLoss):
    """
    Implement the segmentation loss that is the sum of the binary
    cross-entropy and soft inverse Dice loss.
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        """
        super(BCEAndSoftDiceLoss, self).__init__()
        self.bce = BinCrossEntropySegmLoss()
        self.softdice = BinSoftInvDiceLoss(smooth=smooth)

    def forward(self,
                pred_m,
                true_m,
                gate=None
                ):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param gate: (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels) OR None.
        if not none, each value indicated if to ignore the pixel-annotation (
        value=0) or not (value=1). This is an abstention gate.
        :return vector of size (n) contains the binary cross entropy and
        soft inverse Dice loss. values are in [0, +inf[ where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        if gate is not None:
            assert gate.ndim == 2, "'gate.ndim' = {}. must be {}.".format(
                gate.ndim, 2)

            msg = "size mismatches: {}, {}.".format(true_m.shape, pred_m.shape)
            assert true_m.shape == gate.shape, msg

            msg = "'gate' dtype required is torch.float. found {}.".format(
                gate.dtype)
            assert gate.dtype == torch.float, msg

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        return self.bce(
            pred_m=pred_m, true_m=true_m, gate=gate) + self.softdice(
            pred_m=pred_m, true_m=true_m, gate=gate)

    def __str__(self):
        return "{}(): Binary cross-entropy loss  + soft inverse Dice index " \
               "loss for binary segmentation.".format(self.__class__.__name__)


class GetAreaOfMask(nn.Module):
    """
    Compute the active area of a mask.
    """
    def __init__(self, binarize=False):
        """
        Init. function
        :param binarize: if true, we binarize the mask before computing the
        active area.
        """
        super(GetAreaOfMask, self).__init__()

        msg = "'binarize' must be of type bool. found {}.".format(
            type(binarize))
        assert isinstance(binarize, bool), msg

        self.binarize = binarize

    def forward(self, mask, binarize=None):
        """
        Forward function.
        Computes the active area of a mask (~ l1 norm). the mask is supposed
        to be all positive (>= 0.).
        :param mask: pytorch tensor (1, h, w).
        :param binarize: bool. allows to overwrite self.binarize.
        :return: scalar value that is the area.
        """
        assert mask.ndim == 3, "'mask'.ndim must be 3. found {}.".format(
            mask.ndim)
        assert mask.shape[0] == 1, "'mask'.shape[0] must be 1. " \
                                   "found {}.".format(mask.shape[0])
        assert isinstance(binarize, bool), "'bianrize' mut be of type bool." \
                                           "found {}.".format(type(binarize))
        if binarize is None:
            binarize = self.binarize

        if binarize:
            return (((mask >= 0.5) * 1.).float()).sum()
        else:
            return mask.sum()


class HybridLoss(_Loss):
    """
    Implement a hybrid loss for: classification and segmentation tasks.
    """
    def __init__(self,
                 segloss_l=constants.BCEAndSoftDiceLoss,
                 segloss_pl=constants.BCEAndSoftDiceLoss,
                 smooth=1.,
                 elbon=True,
                 init_t=1.,
                 max_t=10.,
                 mulcoef=1.01,
                 subtask=constants.SUBCLSEG,
                 scale_cl=1.,
                 scale_seg=1.,
                 scale_seg_u=1.,
                 scale_seg_u_end=0.001,
                 scale_seg_u_sigma=100.,
                 scale_seg_u_sch=constants.ConstantWeight,
                 max_epochs=1000,
                 scale_seg_u_p_abstention=0.0,
                 freeze_classifier=False,
                 weight_pl=1.
                 ):
        """
        Init. function.
        :param segloss_l: str. valid name of a segmentation loss for the
        labeled samples.
        :param segloss_pl: str. valid name of a segmentation loss for the
        pseudo-labeled samples.
        :param smooth: float. smoothing value for some losses.
        must be  float > 0.
        :param elb: bool. if true, the pseudo-labeled masks are optimized
        using extended log-barrier loss. else, they are directly optimized as
        they are true masks (no uncertainty).
        :param init_t: float > 0. The initial value of `t`. for elb loss.
        :param max_t: float > 0. The maximum allowed value of `t`. for elb loss.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef. for elb loss.
        :param subtask: str. subtask.
        :param scale_cl: float positive scalar. how much to scale the
        classification loss.
        :param scale_seg: float positive scalar. how much to scale the
        supervised segmentation loss.
        :param scale_seg_u: float positive scalar. how much to scale the
        unsupervised segmentation loss.
        :param scale_seg_u_end: float. the final (or minimum allowed value)
        value of scale_seg_u.
        :param scale_seg_u_sigma: float (> 0). scale value for the schedule
        exp for the scale_seg_u.
        :param scale_seg_u_sch: str. a decay schedule for the scale_seg_u.
        :param scale_seg_u_p_abstention: float in [0, 1[. probability of a
        pixel-pseudo-label to be ignored (abstention).
        :param freeze_classifier: bool. if true, the classification loss is
        not considered.
        :param weight_pl: float. value to multiply the loss of the
        pseudo-labeled segmentation loss. must be > 0.
        """
        super(HybridLoss, self).__init__()

        self.freeze_classifier = freeze_classifier
        # classification loss.
        self.cl = CE()
        # segmentation loss
        self.seg_l = self.instantiate_segloss(segloss=segloss_l, smooth=smooth)
        self.seg_pl = self.instantiate_segloss(
            segloss=segloss_pl, smooth=smooth)

        # elb
        self.elbon = elbon
        self.elb = None
        self.t_tracker = []  # track `t` of ELB if there is any.
        if elbon:
            self.elb = _LossExtendedLB(
                init_t=init_t, max_t=max_t, mulcoef=mulcoef)

        self.register_buffer(
            "zero", torch.tensor([0.], requires_grad=False).float())
        self.register_buffer(
            "scale_cl", torch.tensor([scale_cl], requires_grad=False).float())
        self.register_buffer(
            "scale_seg", torch.tensor([scale_seg], requires_grad=False).float())
        self.register_buffer(
            "scale_seg_u", torch.tensor(
                [scale_seg_u], requires_grad=False).float())
        self.subtask = subtask

        msg = "'weight_pl' must be float. found {}.".format(type(weight_pl))
        assert isinstance(weight_pl, float), msg

        msg = "'weight_pl' must be > 0. found {}.".format(weight_pl)
        assert weight_pl > 0., msg
        self.register_buffer(
            "weight_pl", torch.tensor([weight_pl], requires_grad=False).float())

        # callbacks to dynamically update the scales.

        self.cb_scale_cl = None
        self.cb_scale_seg = None
        self.cb_scale_seg_u = self.instantiate_cb_scale(
            scale_seg_u, scale_seg_u_end, scale_seg_u_sigma, max_epochs,
            scale_seg_u_sch
        )

        # initiate scales.
        self.update_scales()

        msg = "'scale_seg_u_p_abstention' must be in [0., 1.]. " \
              "found {}.".format(scale_seg_u_p_abstention)
        assert 0. <= scale_seg_u_p_abstention <= 1., msg
        self.scale_seg_u_p_abstention = scale_seg_u_p_abstention
        self.gate = bernoulli.Bernoulli(
            probs=torch.tensor([1. - scale_seg_u_p_abstention]))

    def set_weight_pl(self, val):
        """
        Set the value of the buffer self.weight_pl.
        This allows more flexibility.
        :param val: float, > 0.
        """
        msg = "'val' must be float. found {}.".format(type(val))
        assert isinstance(val, float), msg

        msg = "'val' must be > 0. found {}.".format(val)
        assert val > 0., msg

        val = torch.tensor([val], requires_grad=False).float()
        # update the buffer that has an initial value (default).
        # keep the buffer on the same device as the initial value.
        self.weight_pl = torch.tensor(
            [val], dtype=torch.float, requires_grad=False,
            device=self.weight_pl.device
        )

    def instantiate_cb_scale(self,
                             scale_seg_u,
                             scale_seg_u_end,
                             scale_seg_u_sigma,
                             max_epochs,
                             scale_seg_u_sch
                             ):
        """
        Instantiate a callback for the scale.
        """
        msg = "Unknown scaling schedule {}. valid {}.".format(
            scale_seg_u_sch, constants.scale_decay)
        assert scale_seg_u_sch in constants.scale_decay, msg

        return decay.__dict__[scale_seg_u_sch](init_val=scale_seg_u,
                                               end_val=scale_seg_u_end,
                                               max_epochs=max_epochs,
                                               sigma=scale_seg_u_sigma
                                               )

    def update_t(self):
        """
        Update the value of `t` of the ELB method.
        :return:
        """
        if self.elb is not None:
            self.t_tracker.append(self.elb.t_lb.item())
            self.elb.update_t()

    def update_scales(self):
        """
        Update scale_cl, scale_seg, scale_seg_u.
        """
        # a buffer can be updated using a non-buffer-tensor. the buffer
        # remains a buffer and its value is the new assigned value.
        # a buffer can not be updated by a float. an error will be raised.

        if self.cb_scale_seg_u is not None:
            self.scale_seg_u = torch.tensor(
                [self.cb_scale_seg_u()], dtype=torch.float, requires_grad=False,
                device=self.scale_seg_u.device
            )

        if self.cb_scale_seg is not None:
            self.scale_seg = torch.tensor(
                [self.cb_scale_seg()], dtype=torch.float, requires_grad=False,
                device=self.scale_seg.device
            )

        if self.cb_scale_cl is not None:
            self.scale_cl = torch.tensor(
                [self.cb_scale_cl()], dtype=torch.float, requires_grad=False,
                device=self.scale_cl.device
            )

    def get_t(self):
        """
        Returns the value of 't_lb' of the ELB method.
        """
        if self.elb is not None:
            return self.elb.get_t()
        else:
            return self.zero

    def instantiate_segloss(self, segloss, smooth):
        """
        Instantiate a segmentation loss.
        :param segloss: str. valid segmentation loss name.
        :return: instance of a segmentation loss.
        """
        msg = "segloss: '{}' is not valid: {}.".format(
            segloss, constants.seglosses)
        assert segloss in constants.seglosses, msg
        return sys.modules[__name__].__dict__[segloss](smooth=smooth)

    def forward(self,
                scores,
                labels,
                targets,
                masks_pred,
                masks_trg,
                tags,
                weights=None,
                avg=True
                ):
        """
        Forward function.
        :param scores: torch tensor (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :param targets: torch tensor  (n, d). used only for our method for
        classification task.
        :param masks_pred: torch tensor. predicted masks (for seg.).
        normalized scores. shape: (n, m) where n is the batch size and m is
        the number of pixels in the mask.
        :param masks_trg: torch tensor. target mask (for seg). shape: (n, m)
        where n is the batch size and m is the number of pixels in the mask.
        :param tags: vector of Log integers. The tag of each sample.
        :param weights: vector of weights. a weight per sample.
        :param avg: bool. if true, the loss is averaged otherwise,
        it is summed up.
        :return: total_loss, cl_loss, seg_l_loss, seg_lp_loss: scalars of the
        total loss, the classification loss, the segmentation loss over full
        supervised samples, the segmentation loss over the pseudo-labeled
        samples.
        """
        if scores is not None:
            device = scores.device
        elif masks_pred is not None:
            device = masks_pred.device
        else:
            raise ValueError("Unable to determine the device")

        cl_loss = torch.tensor([0.],
                               requires_grad=True,
                               device=device,
                               dtype=torch.float
                               )
        seg_l_loss = torch.tensor([0.],
                                  requires_grad=True,
                                  device=device,
                                  dtype=torch.float
                                  )
        seg_lp_loss = torch.tensor([0.],
                                   requires_grad=True,
                                   device=device,
                                   dtype=torch.float
                                   )

        total_loss = torch.tensor([0.],
                                  requires_grad=True,
                                  device=device,
                                  dtype=torch.float
                                  )

        cl, segl, segpl = False, False, False

        # classification
        if (self.subtask in [constants.SUBCL, constants.SUBCLSEG]) and (
            not self.freeze_classifier
        ):
            cl_loss = self.cl(scores=scores, labels=labels, avg=avg)
            cl = True

        # segmentation.
        bsz = float(masks_pred.shape[0])

        if self.subtask in [constants.SUBSEG, constants.SUBCLSEG]:
            # segmentation on supervised samples.
            indx_l = (tags == constants.L).nonzero().squeeze()

            if indx_l.numel() > 0:
                masks_pred_l = masks_pred[indx_l]
                masks_trg_l = masks_trg[indx_l]
                if indx_l.numel() == 1:
                    masks_pred_l = masks_pred_l.unsqueeze(0)
                    masks_trg_l = masks_trg_l.unsqueeze(0)

                seg_l_loss = self.seg_l(pred_m=masks_pred_l, true_m=masks_trg_l)

                if avg:
                    seg_l_loss = seg_l_loss.sum() / bsz
                else:
                    seg_l_loss = seg_l_loss.sum()

                segl = True

            # segmentation on pseudo-labeled samples.
            indx_lp = (tags == constants.PL).nonzero().squeeze()

            if indx_lp.numel() > 0:
                masks_pred_lp = masks_pred[indx_lp]
                masks_trg_lp = masks_trg[indx_lp]
                if indx_lp.numel() == 1:
                    masks_pred_lp = masks_pred_lp.unsqueeze(0)
                    masks_trg_lp = masks_trg_lp.unsqueeze(0)

                # apply the gate (stochastic abstention) only if: 1. it is on.
                # 2. we are in train mode.
                gate = None  # binary. 1. consider the pixel. 0. ignore the
                # pixel.
                if (self.scale_seg_u_p_abstention > 0.) and self.training:
                    # the sampling operation is under torch.no_grad().
                    gate = self.gate.sample(sample_shape=masks_trg_lp.shape)
                    if gate.ndim == 3:
                        gate = gate.squeeze(dim=-1)  # remove the last dim
                        # added by the bern. dist.
                    gate = gate.to(masks_pred_lp.device)  # move to
                    # corresponding device

                seg_lp_loss = self.seg_pl(pred_m=masks_pred_lp,
                                          true_m=masks_trg_lp,
                                          gate=gate
                                          )
                if self.elbon:
                    seg_lp_loss = self.elb(seg_lp_loss)  # summed up loss.
                    if avg:
                        seg_lp_loss = seg_lp_loss / bsz
                elif avg:
                    seg_lp_loss = seg_lp_loss.sum() / bsz

                segpl = True

        # cl. term
        if (self.scale_cl != 0.) and cl:
            total_loss = total_loss + self.scale_cl * cl_loss

        # seg. l. term
        if (self.scale_seg != 0.) and segl:
            total_loss = total_loss + self.scale_seg * seg_l_loss

        # seg. p.pl. term
        ct = self.scale_seg_u * self.weight_pl
        if (ct != 0.) and segpl:
            total_loss = total_loss + ct * seg_lp_loss

        return total_loss, cl_loss, seg_l_loss, seg_lp_loss

    def __str__(self):
        if self.elb is not None:
            str_ = "{}(): Hybdrid loss. CE: {}. SegL: {}. SegPL: {}." \
                   "ELB: {}." \
                   "init_t={}, max_t={}, mulcoef={}".format(
                    self.__class__.__name__, self.cl, self.seg_l, self.seg_pl,
                    self.elbon, self.elb.init_t, self.elb.max_t,
                    self.elb.mulcoef
                    )
        else:
            str_ = "{}(): Hybdrid loss. CE: {}. SegL: {}. SegPL: {}." \
                   "ELB: {}.".format(
                    self.__class__.__name__, self.cl, self.seg_l, self.seg_pl,
                    self.elbon
                    )
        return str_


# ==============================================================================
#                                 METRICS
#                1. ACC: Classification accuracy per view. in [0, 1]. 1 is the
#                best.
# ==============================================================================


class Metrics(nn.Module):
    """
    Compute some metrics.

    1. ACC: Classification accuracy. in [0, 1]. 1 is the best. [if avg=True].
    2. Dice index.
    3. mIOU: mean intersection over union.

    Note: 2 and 3 are for binary segmentation.
    """
    def __init__(self, threshold=0.5):
        """
        Init. function.
        :param threshold: float. threshold in [0., 1.].
        """
        super(Metrics, self).__init__()
        msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
        assert 0 <= threshold <= 1., msg
        msg = "'threshold' type must be float. found {}.".format(
            type(threshold))
        assert isinstance(threshold, float), msg

        self.threshold = threshold
        self.dice = Dice()
        self.iou = IOU()

    def forward(self,
                scores,
                labels,
                tr_loss,
                masks_pred,
                masks_trg,
                avg=False,
                threshold=None,
                ignore_dice=None
                ):
        """
        The forward function.

        :param scores: matrix (n, nbr_c) of unormalized-scores or probabilities.
        :param labels: vector of Log integers. The ground truth labels.
        :param tr_loss: instance of the training loss.
        We use it to compute the predicted label.
        :param masks_pred: torch tensor. predicted masks (for seg.).
        normalized scores. shape: (n, m) where n is the batch size.
        :param masks_trg: torch tensor. target mask (for seg). shape: (n, m)
        where n is the batch size and m is the number of pixels in the mask.
        :param avg: bool If True, the metrics are averaged
        by dividing by the total number of samples.
        :param threshold: float. threshold in [0., 1.] or None. if None,
        we use self.threshold. otherwise, we us this threshold.
        :param ignore_dice: None or torch array of binary values where 1 means
        ignore this sample when computing dice. size (n) where n is the batch 
        size.
        :return:
            acc: scalar (torch.tensor of size 1). classification
            accuracy (avg or sum).
            dice_index: scalar (torch.tensor of size 1). Dice index (avg or
            sum).
            iou: scalar (torch.tensor of size 1). Mean IOU over classes (
            binary) [sum or average over samples].
        """
        msg = "`scores` must be a matrix with size (h, w) where `h` is the " \
              "number of samples, and `w` is the number of classes. We found," \
              " `scores.ndim`={}, and `inputs.shape`={} .... " \
              "[NOT OK]".format(scores.ndim, scores.shape)
        assert scores.ndim == 2, msg
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        assert labels.ndim == 1, "`labels` must be a vector ....[NOT OK]"
        msg = "`labels` and `scores` dimension mismatch....[NOT OK]"
        assert labels.numel() == scores.shape[0], msg

        msg = "'masks_pred.ndim' = {}. must be {}.".format(masks_pred.ndim, 2)
        assert masks_pred.ndim == 2, msg

        msg = "'masks_trg.ndim' = {}. must be {}.".format(masks_trg.ndim, 2)
        assert masks_trg.ndim == 2, msg

        msg = "size mismatches: {}, {}.".format(
            masks_trg.shape, masks_pred.shape
        )
        assert masks_trg.shape == masks_pred.shape, msg
        msg = "'masks_pred' dtype required is torch.float. found {}.".format(
            masks_pred.dtype)
        assert masks_pred.dtype == torch.float, msg
        msg = "'masks_trg' dtype required is torch.float. found {}.".format(
            masks_trg.dtype)
        assert masks_trg.dtype == torch.float, msg

        n, c = scores.shape
        msg = "batch size mismatches. scores {}, masks_pred {}, " \
              "masks_trg {}".format(n, masks_pred.shape[0], masks_trg.shape[0])
        assert n == masks_pred.shape[0] == masks_trg.shape[0], msg

        if ignore_dice is not None:
            assert ignore_dice.dtype == torch.float, "dtype error."

            msg = "ndim must be 1 found {}".format(ignore_dice.ndim)
            assert ignore_dice.ndim == 1, msg
            msg = "number of element must be {}. found {}".format(
                n, ignore_dice.numel())
            assert ignore_dice.numel() == n, msg

            qnt = (ignore_dice == 0.).float().sum()
            qnt += (ignore_dice == 1.).float().sum()
            assert qnt == n, "values must be binary."

        cur_threshold = self.threshold

        if threshold is not None:
            msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
            assert 0. <= threshold <= 1., msg
            msg = "'threshold' type must be float. found {}.".format(
                type(threshold))
            assert isinstance(threshold, float), msg
            cur_threshold = threshold

        # This class should not be included in any gradient computation.
        with torch.no_grad():
            plabels = tr_loss.predict_label(scores)  # predicted labels
            # 1. ACC in [0, 1]
            acc = ((plabels - labels) == 0.).float().sum()

            # 2. Dice index in [0, 1]
            ppixels = self.get_binary_mask(pred_m=masks_pred,
                                           threshold=cur_threshold
                                           )

            dice_index = self.dice(pred_m=ppixels, true_m=masks_trg)

            if ignore_dice is not None:
                dice_index = dice_index * (1. - ignore_dice.view(
                    dice_index.shape))

            dice_index = dice_index.sum()

            # 3. mIOU:
            # foreground
            iou_fgr = self.iou(pred_m=ppixels, true_m=masks_trg)
            # background
            iou_bgr = self.iou(pred_m=1.-ppixels, true_m=1-masks_trg)
            iou = (iou_fgr + iou_bgr) / 2.  # avg. over classes (2)

            if ignore_dice is not None:
                iou = iou * (1. - ignore_dice.view(iou.shape))

            iou = iou.sum()

            if avg:
                acc = acc / float(n)

                # dice na diou.
                if ignore_dice is not None:
                    n = n - (ignore_dice == 1.).float().sum()
                    assert n >= 0, 'ERROR. n={}'.format(n)

                if n != 0:
                    dice_index = dice_index / float(n)
                    iou = iou / float(n)
                else:
                    iou = 0.
                    dice_index = 0.

        return acc, dice_index, iou

    def binarize_mask(self, masks_pred, threshold):
        """
        Predict the binary mask for segmentation.

        :param masks_pred: tensor (n, whatever-dims) of normalized-scores.
        :param threshold: float. threshold in [0., 1.]
        :return: tensor of same shape as `masks_pred`. Contains the binary
        mask, thresholded at `threshold`. dtype: float.
        """
        msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
        assert 0. <= threshold <= 1., msg
        msg = "'threshold' type must be float. found {}.".format(
            type(threshold))
        assert isinstance(threshold, float), msg


        return (masks_pred >= threshold).float()

    def get_binary_mask(self, pred_m, threshold=None):
        """
        Get binary mask by thresholding.
        :param pred_m: torch tensor of shape (n, what-ever-dim)
        :param threshold: float. threshold in [0., 1.] or None. if None,
        we use self.threshold. otherwise, we us this threshold.
        :return:
        """
        cur_threshold = self.threshold

        if threshold is not None:
            msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
            assert 0. <= threshold <= 1., msg
            msg = "'threshold' type must be float. found {}.".format(
                type(threshold))
            assert isinstance(threshold, float), msg
            cur_threshold = threshold

        return self.binarize_mask(pred_m, threshold=cur_threshold)


    def __str__(self):
        return "{}(): computes ACC, Dice index metrics.".format(
            self.__class__.__name__)


# ==============================================================================
#                                 TEST
# ==============================================================================


def test_CE():
    reset_seed(0, check_cudnn=False)
    for weighted in [True, False]:
        instance = CE()
        announce_msg("Testing {}".format(instance))
        announce_msg("weighted: {}".format(weighted))

        cuda = 0
        DEVICE = torch.device(
            "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(cuda))
        instance.to(DEVICE)
        num_classes = 5
        b = 16
        scores = (torch.rand(b, num_classes)).to(DEVICE)
        labels = torch.randint(low=0, high=num_classes, size=(b,),
                               dtype=torch.long
                               ).to(DEVICE)
        tags = torch.randint(low=0, high=3, size=(b,), dtype=torch.long
                             ).to(DEVICE)
        weights = None
        if weighted:
            weights = torch.rand(size=(b,), dtype=torch.float32).to(DEVICE)
        cen = instance(scores=scores, labels=labels, weights=weights, tags=tags)
        print("H(p, q): {}".format(cen))


def test_KL():
    reset_seed(0, check_cudnn=False)

    instance = KL()
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    num_classes = 5
    b = 16
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,),
                           dtype=torch.long
                           ).to(DEVICE)
    targets = torch.rand((b, num_classes), dtype=torch.float32).to(DEVICE)
    targets = F.softmax(targets, dim=1)

    loss = instance(scores=scores, labels=labels, targets=targets)
    print("KL(p, q) with targets: {}".format(loss))

    loss = instance(scores=scores, labels=labels, targets=None)
    print("CE(p, q) with targets=None: {}".format(loss))


def test__JSD():
    reset_seed(0, check_cudnn=False)

    instance = _JSD()
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    num_classes = 5
    b = 16
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,),
                           dtype=torch.long
                           ).to(DEVICE)
    source = torch.rand((b, num_classes), dtype=torch.float32).to(DEVICE)
    source = F.softmax(source, dim=1)

    targets = torch.rand((b, num_classes), dtype=torch.float32).to(DEVICE)
    targets = F.softmax(targets, dim=1)

    loss_p_q = instance(source, targets)
    print("JSD(p, q) with targets: {}".format(loss_p_q))
    loss_q_p = instance(source, targets)
    print("JSD(q, p) with targets: {}".format(loss_q_p))

    print("Diff: {}".format(loss_p_q - loss_q_p))

def test_Dice():
    reset_seed(0, check_cudnn=False)

    instance = Dice()
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 1
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    loss = instance(pred_m.view(b, -1), true_m.view(b, -1))
    print("Dice index: {}".format(loss))
    print(loss.shape)


def test_IOU():
    reset_seed(0, check_cudnn=False)

    instance = IOU()
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 1
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    loss = instance(pred_m.view(b, -1), true_m.view(b, -1))
    print("IOU: {}".format(loss))
    print(loss.shape)


def test_BinSoftInvDiceLoss():
    reset_seed(0, check_cudnn=False)

    instance = BinSoftInvDiceLoss(smooth=1.)
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 10
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    # no abstention.
    probs = 0.0
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), )
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))
    print(loss.shape)
    probs = 0.9
    gater = bernoulli.Bernoulli(probs=torch.tensor([1. - probs]))
    gate = gater.sample(sample_shape=true_m.view(b, -1).shape).squeeze(dim=-1)
    gate = gate.to(DEVICE)
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), gate)
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))


def test_BinL1SegmLoss():
    reset_seed(0, check_cudnn=False)

    instance = BinL1SegmLoss()
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 10
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    # no abstention.
    probs = 0.0
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), )
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))
    print(loss.shape)
    probs = 0.9
    gater = bernoulli.Bernoulli(probs=torch.tensor([1. - probs]))
    gate = gater.sample(sample_shape=true_m.view(b, -1).shape).squeeze(dim=-1)
    gate = gate.to(DEVICE)
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), gate)
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))


def test_BinIOUSegmLoss():
    reset_seed(0, check_cudnn=False)

    instance = BinIOUSegmLoss(smooth=1.)
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 10
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    # no abstention.
    probs = 0.0
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), )
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))
    print(loss.shape)
    probs = 0.9
    gater = bernoulli.Bernoulli(probs=torch.tensor([1. - probs]))
    gate = gater.sample(sample_shape=true_m.view(b, -1).shape).squeeze(dim=-1)
    gate = gate.to(DEVICE)
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), gate)
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))


def test_BinL1SegAndIOUSegmLoss():
    reset_seed(0, check_cudnn=False)

    instance = BinL1SegAndIOUSegmLoss(smooth=1.)
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 10
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    # no abstention.
    probs = 0.0
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), )
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))
    print(loss.shape)
    probs = 0.9
    gater = bernoulli.Bernoulli(probs=torch.tensor([1. - probs]))
    gate = gater.sample(sample_shape=true_m.view(b, -1).shape).squeeze(dim=-1)
    gate = gate.to(DEVICE)
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), gate)
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))


def test_BinCrossEntropySegmLoss():
    reset_seed(0, check_cudnn=False)

    instance = BinCrossEntropySegmLoss()
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 10
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    # no abstention.
    probs = 0.0
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), )
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))
    print(loss.shape)
    probs = 0.9
    gater = bernoulli.Bernoulli(probs=torch.tensor([1. - probs]))
    gate = gater.sample(sample_shape=true_m.view(b, -1).shape).squeeze(dim=-1)
    gate = gate.to(DEVICE)
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), gate)
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))


def test_BCEAndSoftDiceLoss():
    reset_seed(0, check_cudnn=False)

    instance = BCEAndSoftDiceLoss(smooth=1.)
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    h, w = 32, 64
    b = 10
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    # no abstention.
    probs = 0.0
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), )
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))
    print(loss.shape)
    probs = 0.9
    gater = bernoulli.Bernoulli(probs=torch.tensor([1. - probs]))
    gate = gater.sample(sample_shape=true_m.view(b, -1).shape).squeeze(dim=-1)
    gate = gate.to(DEVICE)
    loss = instance(pred_m.view(b, -1), true_m.view(b, -1), gate)
    print("Props = {}. Loss value: {}".format(probs, loss))
    print("sum loss {}".format(loss.sum()))


def test_HybridLoss():
    reset_seed(0, check_cudnn=False)

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))

    num_classes = 10
    h, w = 32, 64
    b = 10
    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,),
                           dtype=torch.long
                           ).to(DEVICE)
    tags = torch.randint(low=0, high=3, size=(b,), dtype=torch.long
                         ).to(DEVICE)
    init_t = 1.
    max_t = 10.
    mulcoef = 1.01

    for elbon in [False, True]:
        for segloss in constants.seglosses:
            instance = HybridLoss(segloss=segloss, smooth=1., elbon=elbon,
                                  init_t=init_t, max_t=max_t, mulcoef=mulcoef)
            announce_msg("Testing {}".format(instance))
            instance.to(DEVICE)
            losses = instance(scores, labels, targets=None,
                              masks_pred=pred_m.view(b, -1),
                              masks_trg=true_m.view(b, -1), tags=tags,
                              weights=None,  avg=True)
            print("Losses: total {}, cl {}, segl {}, seglp {}".format(
                losses[0], losses[1], losses[2], losses[3]))
            print(losses[0].shape)
            if elbon:
                print("t of elb: {} before up".format(instance.get_t()))
                instance.update_t()
                print("t of elb after up: {}".format(instance.get_t()))


def test_GetAreaOfMask():
    reset_seed(0, check_cudnn=False)
    instance = GetAreaOfMask(binarize=True)
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)

    mask = (torch.sigmoid(torch.rand(1, 10, 20))).to(DEVICE)

    out = instance(mask)

    print("ACTIVE AREA: {}".format(out))


def test__LossExtendedLB():
    reset_seed(0, check_cudnn=False)
    instance = _LossExtendedLB(init_t=1., max_t=10., mulcoef=1.01)
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)

    b = 16
    fx = (torch.rand(b)).to(DEVICE)

    out = instance(fx)
    for r in range(10):
        instance.update_t()
        print("epoch {}. t: {}.".format(r, instance.t_lb))
    print("Loss ELB.sum(): {}".format(out))


def test_LossHistogramsMatching():
    """
    Test: LossHistogramsMatching().
    :return:
    """
    reset_seed(0, check_cudnn=False)
    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    bins = 256
    min = 0.
    max = 1.
    sigma = 1e5
    m = LossHistogramsMatching()
    m.to(DEVICE)

    batchs, bins = 2, 256
    src = torch.rand((batchs, bins)).to(DEVICE)
    src = src / src.sum(dim=1).unsqueeze(1)
    trg = torch.rand((batchs, bins)).to(DEVICE)
    trg = trg / trg.sum(dim=1).unsqueeze(1)

    out = m(trg, src)
    print(out.shape)
    print(out)


def test_Metrics():
    # TODO
    reset_seed(0, check_cudnn=False)
    instance = Metrics()
    announce_msg("Testing {}".format(instance))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)
    num_classes = 4
    b = 100
    h, w = 32, 64
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    pred_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)
    true_m = torch.randint(low=0, high=2, size=(b, 1, h, w),
                           dtype=torch.float
                           ).to(DEVICE)

    tr_loss = CE()
    for avg in [True, False]:
        mtrs = instance(scores, labels, tr_loss, pred_m.view(b, -1),
                       true_m.view(b, -1), avg=avg)
        print("ACC: {} -- avg: {}".format(mtrs[0].item(), avg))
        print("Dice: {} -- avg: {}".format(mtrs[1].item(), avg))
        print("MIOU: {} -- avg: {}".format(mtrs[2].item(), avg))


def test_all():
    test_CE()
    test_KL()
    test_Metrics()


if __name__ == "__main__":
    # test_LossHistogramsMatching()
    # test___LossTotalVariationMask()
    # test_CE()
    # test__JSD()
    # test_Dice()
    # test_IOU()
    # test_BinSoftInvDiceLoss()
    # test_BinCrossEntropySegmLoss()
    # test_BCEAndSoftDiceLoss()
    # test_BinL1SegmLoss()
    # test_BinIOUSegmLoss()
    # test_BinL1SegAndIOUSegmLoss()
    # test_HybridLoss()
    # test_KL()
    test_Metrics()

    # test__LossExtendedLB()
    # test___LossInequalityBounds()
    # test_GetAreaOfMask()

    # test_all()
