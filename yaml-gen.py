#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""yaml-generate-configs.py
Generates yaml configurations for experiments.
"""


from os.path import join

import yaml
from PIL import Image

import constants


# DEFAULT configuration.
config = {
    # ==========================================================================
    #                               GENERAL STUFF
    # ==========================================================================
    "MYSEED": 0,  # Seed for reproducibility. int >= 0.
    "cudaid": 0,  # int. cudaid.
    "debug_subfolder": '',  # subfolder used for debug. if '', we do not
    # consider it.
    "dataset": "mnist",  # name of the dataset:
    # task CL: 'cifar-10', 'cifar-100', 'svhn', 'mnist'.
    # task SEG: 'glas'
    "name_classes": 'encoding.yaml',
    # dict. name classes and corresponding int. If dict it too big,
    # you can dump it in the fold folder in a yaml file.
    # We will load it when needed. Use the name of the basename of the file.
    # example: 'encoding.yaml'
    "num_classes": 10,  # Total number of classes.
    "num_masks": 1,  # number of masks to produce. supports only: 1.
    "split": 0,  # split id.
    "fold": 0,  # folder id.
    "fold_folder": "./folds",  # relative path to the folder of the folds.
    "resize": None,  # PIL format of the image size (w, h). operates only on
    # the image and not the mask unless `resize_mask` is explicitly set to true.
    "resize_h_to": 128,  # int or None. resize the original image height into
    # this value. the width will be computed accordingly to preserve the
    # proportions. this is an alternative to 'resize'. they can not be both
    # set. they are exclusive. operates only on the image and not the mask
    # unless `resize_mask` is explicitly set to true.
    "resize_mask": False,  # if true, the original mask of the original image
    # is resized to `resize` or `resize_h_to`,
    # The size to which the original images are resized to.
    "crop_size": None,  # int, or tuple (h, w). Size of the patches to be
    # cropped (h, w). or
    # None if you want to use the entire image. This must mean that all the
    # images have the same size otherwise an error will be raised.
    "ratio_scale_patch": 1.,  # the ratio to which the cropped patch is scaled.
    # during evaluation, original images are also rescaled using this ratio.
    # if you want to keep the cropped patch as it is, set this variable to 1.
    "up_scale_small_dim_to": None,  # int or None. Upscale only images that
    # have the min-dimension is lower than this variable.
    # If int, the images are upscaled to this size while preserving the
    # ratio. See loader.PhotoDataset().
    "scale_algo": Image.LANCZOS,  # int, resize algorithm. Possible choices:
    # Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING,
    # Image.BICUBIC, Image.LANCZOS. See: https://pillow.readthedocs.io/en/
    # stable/reference/Image.html#PIL.Image.Image.resize
    "padding_ratio": None,  # or 0.5 for example,  # padding ratio for the
    # original image for (top/bottom) and (left/right). Can be applied on both,
    # training/evaluation modes. To be specified in PhotoDataset().
    # If specified, only training images are padded. To pad evaluation
    # images, you need to set the variable: `pad_eval` to True.
    "pad_eval": False,  # If True, evaluation images are padded in the same way.
    # The final mask is cropped inside the predicted mask (since this last one
    # is bigger due to the padding).
    "padding_mode": "reflect",  # type of padding. Accepted modes:
    # https://pytorch.org/docs/stable/torchvision/transforms.html#
    # torchvision.transforms.functional.pad
    "batch_size": 8,  # the batch size for training.
    "valid_batch_size": 8,  # the batch size for validation.
    "num_workers": 8,  # number of workers for dataloader of the trainset.
    "max_epochs": 150,  # number of training epochs.
    # ==========================================================================
    #                      VISUALISATION OF REGIONS OF INTEREST
    # ==========================================================================
    "floating": 3,  # the number of floating points to print over the maps.
    "height_tag": 60,  # the height of the margin where the tag is written.
    # ==========================================================================
    #                             OPTIMIZER (n0)
    #                            TRAIN THE MODEL
    # ==========================================================================
    "optimizer": {  # the optimizer
        # ==================== SGD =======================
        "optn0__name_optimizer": "sgd",  # str name. 'sgd', 'adam'
        "optn0__lr": 0.001,  # Initial learning rate.
        "optn0__momentum": 0.9,  # Momentum.
        "optn0__dampening": 0.,  # dampening.
        "optn0__weight_decay": 1e-5,  # The weight decay (L2) over the
        # parameters.
        "optn0__nesterov": True,  # If True, Nesterov algorithm is used.
        # ==================== ADAM =========================
        "optn0__beta1": 0.9,  # beta1.
        "optn0__beta2": 0.999,  # beta2
        "optn0__eps_adam": 1e-08,  # eps. for numerical stability.
        "optn0__amsgrad": False,  # Use amsgrad variant or not.
        # ========== LR scheduler: how to adjust the learning rate. ============
        "optn0__lr_scheduler": True,  # if true, we use a learning rate
        # scheduler.
        # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
        "optn0__name_lr_scheduler": "mystep",  # str name.
        "optn0__step_size": 40,  # Frequency of which to adjust the lr.
        "optn0__gamma": 0.1,  # the update coefficient: lr = gamma * lr.
        "optn0__last_epoch": -1,  # the index of the last epoch where to stop
        # adjusting the LR.
        "optn0__min_lr": 1e-7,  # minimum allowed value for lr.
        "optn0__t_max": 100,  # T_max for cosine schedule.
    },
    # ==========================================================================
    #                              MODEL
    # ==========================================================================
    "model": {
        "name_model": constants.HYBRIDMODEL,  # name of the model.
        # see: constants.nets.
        "backbone": constants.RESNET18,  # backbone for task of SEG.
        "base_width": 24,  # base width of upscale part of U-Net.
        "leak": 64,  # dim of feature maps extracted from the classifier maps to
        # be used for segmentation.
        "backbone_dropout": 0.0,  # dropout of the backbone. used for bayesian
        # al. and other techs.
        "output_stride": 8,  # output stride for deeplab. supported values:
        # 8, 16. the lower the value, the higher the computation.
        "freeze_bn": False,  # freeze deeplab bn or not.
        "path_pre_trained": None,  # None, `None` or a valid str-path. if str,
        # it is the absolute/relative path to the pretrained model. This can
        # be useful to resume training or to force using a filepath to some
        # pretrained weights.
        "pre_trained": False,  # if pretrained or not.
        "strict": True,  # bool. Must be always be True. if True,
        # the pretrained model has to have the exact architecture as this
        # current model. if not, an error will be raise. if False, we do the
        # best. no error will be raised in case of mismatch.
        # =============================  classifier ==========================
        "modalities": 5,  # number of modalities (wildcat).
        "kmax": 0.1,  # kmax. (wildcat)
        "kmin": 0.1,  # kmin. (wildcat)
        "alpha": 0.0,  # alpha. (wildcat)
        "dropout": 0.0,  # dropout over the kmin and kmax selected activations
        # .. (wildcat).
        "dropoutnetssl": 0.5  # dropout used for sota_ssl model.
    },
    # ==========================================================================
    #                          ACTIVE LEARNING
    # ==========================================================================
    "al_it": 0,  # int, the active learning iteration. 0 is the initial
    # iteration.
    "max_al_its": 20,  # maximum al rounds.
    "al_type": constants.AL_RANDOM,  # type of active learning selection. see
    # constants.py
    "clustering": constants.CLUSTER_RANDOM,  # how select
    # samples in our method. see constants.ours_clustering
    "knn": 1,  # k of the k-nn method. it indicates how far we go to search
    # for labeled neighbors. default is 1 (check only the nearest neighbor,
    # if it is labeled, propagate its label). if k > 1: keep searching for
    # the closest labeled sample but do not go beyond k neighbors.
    "mcdropout_t": 50,  # how many sampling to do for the mc-dropout al.
    "loss": constants.HYBRIDLOSS,  # str, training loss.
    "segloss_l": constants.BinCrossEntropySegmLoss,  # segmentation loss for
    # labeled samples.
    "segloss_pl": constants.BinCrossEntropySegmLoss,  # segmentation loss for
    # pseudo-labeled samples.
    "scale_cl": 1.,  # how much to scale the classification loss. it is a way
    # to scale the learning rate.
    "scale_seg": 1.,  # how much to scale the supervised segmentation loss. it
    # is a way to scale the learning rate..
    "scale_seg_u": 1.,  # how much to scale the unsupervised segmentation
    # loss. it is a way to reduce the learning rate. this is the initial value.
    "scale_seg_u_end": 0.001,  # final value of scale_seg_u (allowed/ or
    # final for linear)
    "scale_seg_u_sch": constants.ConstantWeight,  # schedule how to update the
    # scale_seg_u.
    "scale_seg_u_sigma": 200.,   # sigma for ExponentialDecayWeight.
    "scale_seg_u_p_abstention": 0.0,  # probability of a pixel-pseudo-label to
    # be ignored (abstention mechanism). it is similar to the p of dropout,
    # and the effect of dropout. 0.0 indicates to consider all the pixels.
    # a value of 1. means ignore everything. value in [0., 1.].
    "weight_pseudo_loss": False,  # if true, the loss of pseudo-labeled
    # samples is weighted using 1./(number of pseudo-labeled samples).
    "seg_elbon": False,  # if true, the pseudo-segmented samples are considered
    # exactly as the supervised samples. if true, `t` is updated every 1 epoch.
    "seg_init_t": 1.,  # considered when `seg_elbon` is True. used ELB for
    # pseudo-segmented samples loss term.
    "seg_max_t": 10.,  # considered when `seg_elbon` is True. used ELB for
    # pseudo-segmented samples loss term.
    "seg_mulcoef": 1.01,  # considered when `seg_elbon` is True. used ELB for
    # pseudo-segmented samples loss term.
    "seg_smooth": 1.,  # used to smooth Dice loss. considered when Dice loss
    # is used.
    "exp_id": '123456',  # Exp id. a random number that can be passed to create
    # the exp folder. it helps writing in the same folder when doing multiple
    # active learning rounds.
    "p_samples": 2.,  # percentage of samples to label at each active
    # iteration. the number of samples will be: original size of train set *
    # p_samples / 100. this means that the number of samples that we add does
    # not change through active learning rounds.
    "p_init_samples": 2.,  # percentage of samples to select at round 0 as
    # pre-labeled samples. it is computed as for 'p_samples'.
    "task": constants.SEG,  # type of the task: classification, segmentation.
    "subtask": constants.SUBCLSEG,  # subtask. see constans.subtasks. allows
    # the eactivate a task. for example, in the case of weak-sup (SEG, CL),
    # one may want to activate only the classification task without the need
    # to do MAJOR changes in the code. use with caution.
    "pair_w_batch_size": 25000,  # batch size used to compute the pairwise sim.
    "use_dist_global_hist": True,  # if true, the measure of how far 2
    # samples includes the measure of how far their histograms. This must be
    # true.

    # ==========================================================================
    #                               PROPERTIES:
    #                               COMPUTATION
    # ==========================================================================
    "nbr_bins_histc": 256,  # number of bins to create histograms.
    "min_histc": 0.,  # min values in histogram (probability dist.).
    "max_histc": 1.,  # max value in histogram (probability dist.).
    "sigma_histc": 1e5,  # sigma for histogram (probability dist.).
    # the higher the better the approximation. but too high will prevent
    # learning. 1e5 is a good choice.
    "resize_h_to_opt_mask": None,  # int or None. same as 'resize_h_to' but
    # applied only to the dataset used to optimize the mask.
    "resize_mask_opt_mask": True,  # same as 'resize_mask' but applied only
    # to dataset used to optimize mask.
    "protect_pl": True,  # if true, pseudo-labeled samples are not sampled to
    # be labeled unless there are no unlabeled samples left.
    #  ------------------------------------------------------------------
    "freeze_classifier": False,  # if true, the entire classifier is freezed
    # (not trained). this implies that only the trainset will be composed
    # only of pixel-wise labeled and pseudo-labeled samples for the SEG task.
    "estimate_pseg_thres": True,  # if true, the pseudo-masks are obtained
    # using a threshold that is estimated over the validation set. otherwise,
    # a threshold of `seg_threshold` is used.
    "seg_threshold": 0.5,   # segmentation threshold.
    "estimate_seg_thres": True,  # if true, the masks of prediction are
    # computed using a threshold that is estimated over the validation set.
    # if false, `seg_threshold` is used.
    # ==========================================================================
    "smooth_img": True,  # if true, we smooth the image using a gaussian
    # filter. smoothing an image helps removing tiny useless detail.
    "smooth_img_ksz": 5,  # kernel size of the smoothing of the image.
    "smooth_img_sigma": 5.,  # sigma of the gaussian for smoothing the image.
    "enhance_color": True,  # color image is enhanced. (for the dataloader)
    "enhance_color_fact": 3.,  # color enhancemen factor.
    # float [0., [. color enhancing factor. 0. gives black and white.
    # 1 gives the same image. low values than 1
    # means less color, higher values than 1 means more colors.(for the
    # dataloader)
    # ==========================================================================
    #                PROPERTIES/MASK OPTIMIZATION:
    #                           STORING
    # ==========================================================================
    "share_masks": True,  # if this is true, the mask obtained from the pair
    # idu_idl is stored in a folder that is shared among all the active
    # learning rounds to avoid recomputing it again. this saves time. This
    # has to be true because the other way is so not-optimal.
}

# [1]: Understanding Deep Networks via Extremal Perturbations and Smooth Masks.
# https://arxiv.org/abs/1910.08485, 2019.


l_datasets = [
    # cl
    # constants.CIFAR_10,
    # constants.CIFAR_100,
    # constants.SVHN,
    # constants.MNIST,
    # seg
    constants.GLAS,
    constants.CUB,
    constants.OXF,
    constants.CAM16
]


for ds in l_datasets:
    config.update({"dataset": ds})
    # rectify depending on the dataset.
    if config['dataset'] in ['cifar-10', 'cifar-100', 'svhn', 'mnist']:
        config['valid_batch_size'] = 1000

    if config['dataset'] in [constants.GLAS, constants.CUB,
                             constants.OXF, constants.CAM16]:
        config['task'] = constants.SEG
        config['batch_size'] = 8
        config['valid_batch_size'] = 1
        config['pair_w_batch_size'] = 1

        config["model"]["name"] = constants.HYBRIDMODEL
        config['loss'] = constants.HYBRIDLOSS
        config['segloss'] = constants.BinCrossEntropySegmLoss

    if config['dataset'] == constants.CAM16:
        config['pair_w_batch_size'] = 32

    if config["dataset"] == constants.GLAS:
        config['crop_size'] = (416, 416)
        config['up_scale_small_dim_to'] = 432
        config["ratio_scale_patch"] = 0.9
        config['padding_ratio'] = 0.01
        config['pad_eval'] = True
        config["num_classes"] = 2
        config["max_epochs"] = 1000

        # stats images (train, split0, fold 0)
        # MIN H 430, 	 MAX H 522
        # MIN W 567, 	 MAX W 775.
    elif config['dataset'] == constants.CUB:
        config['crop_size'] = (416, 416)
        config['up_scale_small_dim_to'] = 432
        config["ratio_scale_patch"] = 0.9
        config['padding_ratio'] = 0.01
        config['pad_eval'] = True
        config["num_classes"] = 200
        config["max_epochs"] = 100

        # min h 120, 	 max h 500
        # min w 121, 	 max w 500
    elif config['dataset'] == constants.OXF:
        config['crop_size'] = (416, 416)
        config['up_scale_small_dim_to'] = 432
        config["ratio_scale_patch"] = 0.9
        config['padding_ratio'] = 0.01
        config['pad_eval'] = True
        config["num_classes"] = 102
        config["max_epochs"] = 100
        # min h 500, 	 max h 993
        # min w 500, 	 max w 919
    elif config['dataset'] == constants.CAM16:
        config['crop_size'] = (416, 416)
        config['up_scale_small_dim_to'] = None
        config["ratio_scale_patch"] = 0.9
        config['padding_ratio'] = 0.01
        config['pad_eval'] = True
        config["num_classes"] = 2
        config["max_epochs"] = 100
        # min h 512, 	 max h 512
        # min w 512, 	 max w 512

    fold_yaml = "config_yaml"
    fold_bash = "config_bash"
    # name_config = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    name_config = config['dataset']
    name_bash = join(fold_bash, name_config + ".sh")
    name_yaml = join(fold_yaml, name_config + ".yaml")
    print("Generated yaml for: {} dataset in `{}`.".format(ds, name_yaml))


    with open(name_yaml, 'w') as f:
        yaml.dump(config, f)
