"""
Create jobs to run using *.sh files.
- Generates one file that contains all the commands to run one active learning
method over one dataset for ALL the seed.
- Generates one file that contains all the commands to run one active learning
method over one dataset for EACH seed.
"""

import datetime as dt
import os
from os.path import join
from datetime import timedelta
import itertools
import shutil
import textwrap

import numpy as np


from tools import get_rootpath_2_dataset, Dict2Obj


from shared import announce_msg


import constants


def get_combos(llists, namekeys=None):
    """
    Do Cartesian product over list of lists.
    :param llists: list of lists.
    :param namekeys: list of names. each name is associated with a list.
    :return:
    """
    cmds = []
    for combo in itertools.product(*llists):
        partial = ""
        if namekeys is not None:
            for pair in list(zip(namekeys, combo)):
                partial += "--{} {} ".format(pair[0], pair[1])
        else:
            partial = " ".join(combo)
        cmds.append(partial)

    return cmds


def get_opt_model(dataset,
                  epochs=None,
                  subtask=constants.SUBCLSEG,
                  al_type=constants.AL_FULL_SUP
                  ):
    """
    Get the command line by changing the hyperparameters of:
    1. optimizer
    2. model
    according to the dataset.
    :return: list of commands.
    """
    keys = dict()

    # ==========================================================================
    #                                    GLAS
    # ==========================================================================
    if dataset == constants.GLAS:
        # optim.
        keys["optn0__name_optimizer"] = ["sgd"]
        keys["batch_size"] = [20]
        keys["valid_batch_size"] = [1]

        if al_type == constants.AL_WSL:
            keys["optn0__lr"] = [0.0001]
        else:
            keys["optn0__lr"] = [0.1]

        keys["optn0__weight_decay"] = [1e-4]
        keys["optn0__momentum"] = [0.9]
        keys["max_epochs"] = [epochs if epochs is not None else 1000]
        keys["optn0__name_lr_scheduler"] = ["mystep"]  # cosine, mystep.
        keys["optn0__lr_scheduler"] = [True]
        keys["optn0__step_size"] = [100]  # mystep.
        keys["optn0__gamma"] = [0.9]  # mystep.
        keys["optn0__min_lr"] = [1e-7]  # mystep.
        keys["optn0__t_max"] = [50]  # cosine.

        # adam
        keys["optn0__beta1"] = [0.9]
        keys["optn0__beta2"] = [0.999]
        keys["optn0__eps_adam"] = [1e-8]
        keys["optn0__amsgrad"] = [False]


        # model":
        keys["name_model"] = [constants.HYBRIDMODEL]
        keys["backbone"] = [constants.RESNET18]
        keys["backbone_dropout"] = [0.0]
        # keys["output_stride"] = [8]  # 8, or 16.
        # keys["freeze_bn"] = [False]
        if subtask != constants.SUBCL:
            keys["path_pre_trained"] = ['./pretrained/resnet18-glas.pt']
        else:
            keys["path_pre_trained"] = [None]
        keys["pre_trained"] = [True]  # the first initial model (imagenet)

        keys["alpha"] = [0.6]
        keys["kmax"] = [0.1]
        keys["kmin"] = [0.1]
        keys["dropout"] = [0.1]
        keys["dropoutnetssl"] = [0.5]
        keys["modalities"] = [5]

        # data processing
        keys['crop_size'] = [416]
        keys['ratio_scale_patch'] = [.9]
        keys['up_scale_small_dim_to'] = [432]
        keys['padding_ratio'] = [0.01]
        keys['pad_eval'] = [True]

    # ==========================================================================
    #                                    CAMELYON16
    # ==========================================================================
    if dataset == constants.CAM16:
        # optim.
        keys["optn0__name_optimizer"] = ["sgd"]

        if al_type == constants.AL_WSL:
            keys["valid_batch_size"] = [1]
            keys["batch_size"] = [20]
            keys["optn0__lr"] = [0.001]
        else:
            keys["valid_batch_size"] = [16]
            keys["batch_size"] = [16]
            keys["optn0__lr"] = [0.01]

        keys["optn0__weight_decay"] = [1e-4]
        keys["optn0__momentum"] = [0.9]
        if al_type == constants.AL_WSL:
            keys["max_epochs"] = [epochs if epochs is not None else 20]
        else:
            keys["max_epochs"] = [epochs if epochs is not None else 90]

        keys["optn0__name_lr_scheduler"] = ["mystep"]  # cosine, mystep.
        keys["optn0__lr_scheduler"] = [True]
        if al_type == constants.AL_WSL:
            keys["optn0__gamma"] = [0.1]  # mystep.
            keys["optn0__step_size"] = [20]  # mystep.
        else:
            keys["optn0__gamma"] = [0.5]  # mystep.
            keys["optn0__step_size"] = [30]  # mystep.

        keys["optn0__min_lr"] = [1e-7]  # mystep.
        keys["optn0__t_max"] = [50]  # cosine.

        # adam
        keys["optn0__beta1"] = [0.9]
        keys["optn0__beta2"] = [0.999]
        keys["optn0__eps_adam"] = [1e-8]
        keys["optn0__amsgrad"] = [False]

        # model":
        keys["name_model"] = [constants.HYBRIDMODEL]
        keys["backbone"] = [constants.RESNET18]
        keys["backbone_dropout"] = [0.0]
        # keys["output_stride"] = [8]  # 8, or 16.
        # keys["freeze_bn"] = [False]
        if subtask != constants.SUBCL:
            keys["path_pre_trained"] = ['./pretrained/resnet18-camelyon16.pt']
        else:
            keys["path_pre_trained"] = [None]
        keys["pre_trained"] = [True]  # the first initial model (imagenet)

        keys["alpha"] = [0.6]
        keys["kmax"] = [0.1]
        keys["kmin"] = [0.1]
        keys["dropout"] = [0.0]
        keys["dropoutnetssl"] = [0.5]
        keys["modalities"] = [5]

        # data processing
        keys['crop_size'] = [416]
        keys['ratio_scale_patch'] = [.9]
        keys['padding_ratio'] = [0.01]
        keys['pad_eval'] = [True]
    # ==========================================================================
    #                                    CUB
    # ==========================================================================
    if dataset == constants.CUB:
        # optim.
        keys["optn0__name_optimizer"] = ["sgd"]
        keys["batch_size"] = [8]
        keys["valid_batch_size"] = [1]
        if al_type == constants.AL_WSL:
            keys["optn0__lr"] = [0.01]
        else:
            keys["optn0__lr"] = [0.1]
        keys["optn0__weight_decay"] = [1e-4]
        keys["optn0__momentum"] = [0.9]
        if al_type == constants.AL_WSL:
            keys["max_epochs"] = [epochs if epochs is not None else 90]
        else:
            keys["max_epochs"] = [epochs if epochs is not None else 30]

        keys["optn0__name_lr_scheduler"] = ["mystep"]  # cosine, mystep.
        keys["optn0__lr_scheduler"] = [True]
        keys["optn0__step_size"] = [10]  # mystep.
        if al_type == constants.AL_WSL:
            keys["optn0__gamma"] = [0.9]  # mystep.
        else:
            keys["optn0__gamma"] = [0.95]  # mystep.

        keys["optn0__min_lr"] = [1e-7]  # mystep.
        keys["optn0__t_max"] = [50]  # cosine.

        # adam
        keys["optn0__beta1"] = [0.9]
        keys["optn0__beta2"] = [0.999]
        keys["optn0__eps_adam"] = [1e-8]
        keys["optn0__amsgrad"] = [False]

        # model":
        keys["name_model"] = [constants.HYBRIDMODEL]
        keys["backbone"] = [constants.RESNET18]
        keys["backbone_dropout"] = [0.0]
        # keys["output_stride"] = [8]  # 8, or 16.
        # keys["freeze_bn"] = [False]
        if subtask != constants.SUBCL:
            keys["path_pre_trained"] = ['./pretrained/resnet18-cub.pt']
        else:
            keys["path_pre_trained"] = [None]

        keys["pre_trained"] = [True]  # the first initial model (imagenet)

        keys["alpha"] = [0.6]
        keys["kmax"] = [0.1]
        keys["kmin"] = [0.1]
        keys["dropout"] = [0.1]
        keys["dropoutnetssl"] = [0.5]
        keys["modalities"] = [5]

        # data processing
        keys['crop_size'] = [416]
        keys['ratio_scale_patch'] = [.9]
        keys['up_scale_small_dim_to'] = [432]
        keys['padding_ratio'] = [0.01]
        keys['pad_eval'] = [True]

    # ==========================================================================
    #                                    OXF
    # ==========================================================================
    if dataset == constants.OXF:
        # optim.
        keys["optn0__name_optimizer"] = ["sgd"]
        keys["batch_size"] = [8]
        keys["valid_batch_size"] = [1]
        if al_type == constants.AL_WSL:
            keys["optn0__lr"] = [0.01]
        else:
            keys["optn0__lr"] = [0.1]
        keys["optn0__weight_decay"] = [1e-4]
        keys["optn0__momentum"] = [0.9]
        if al_type == constants.AL_WSL:
            keys["max_epochs"] = [epochs if epochs is not None else 90]
        else:
            keys["max_epochs"] = [epochs if epochs is not None else 30]

        keys["optn0__name_lr_scheduler"] = ["mystep"]  # cosine, mystep.
        keys["optn0__lr_scheduler"] = [True]
        keys["optn0__step_size"] = [10]  # mystep.
        if al_type == constants.AL_WSL:
            keys["optn0__gamma"] = [0.9]  # mystep.
        else:
            keys["optn0__gamma"] = [0.95]  # mystep.

        keys["optn0__min_lr"] = [1e-7]  # mystep.
        keys["optn0__t_max"] = [50]  # cosine.

        # adam
        keys["optn0__beta1"] = [0.9]
        keys["optn0__beta2"] = [0.999]
        keys["optn0__eps_adam"] = [1e-8]
        keys["optn0__amsgrad"] = [False]

        # model":
        keys["name_model"] = [constants.HYBRIDMODEL]
        keys["backbone"] = [constants.RESNET18]
        keys["backbone_dropout"] = [0.0]
        # keys["output_stride"] = [8]  # 8, or 16.
        # keys["freeze_bn"] = [False]
        if subtask != constants.SUBCL:
            keys["path_pre_trained"] = ['./pretrained/resnet18-oxf.pt']
        else:
            keys["path_pre_trained"] = [None]

        keys["pre_trained"] = [True]  # the first initial model (imagenet)

        keys["alpha"] = [0.6]
        keys["kmax"] = [0.1]
        keys["kmin"] = [0.1]
        keys["dropout"] = [0.1]
        keys["dropoutnetssl"] = [0.5]
        keys["modalities"] = [5]

        # data processing
        keys['crop_size'] = [416]
        keys['ratio_scale_patch'] = [.9]
        keys['up_scale_small_dim_to'] = [432]
        keys['padding_ratio'] = [0.01]
        keys['pad_eval'] = [True]

    # shared
    if "CC_CLUSTER" in os.environ.keys():
        keys['cudaid'] = [0]
    else:
        keys['cudaid'] = ['$cudaid']

    llists = [keys[k] for k in keys.keys()]
    namekeys = [k for k in keys.keys()]

    return get_combos(llists, namekeys)


def get_al(al_type):
    """
    Get some of active learning config
    :param dog: bool. if true, we store some stuff to see it.
    :return:
    """
    keys = dict()
    # storing
    if al_type != constants.AL_WSL:
        keys["subtask"] = [constants.SUBCLSEG]
    else:
        keys["subtask"] = [constants.SUBCL]


    llists = [keys[k] for k in keys.keys()]
    namekeys = [k for k in keys.keys()]
    cmds = get_combos(llists, namekeys)

    return cmds


def get_optmi_mask():
    """
    Get some of optimization mask configs.
    :return:
    """
    keys = dict()
    keys["nbr_bins_histc"] = [256]
    keys["min_histc"] = [0.]
    keys["max_histc"] = [1.]
    keys["sigma_histc"] = [1e5]

    keys["resize_h_to_opt_mask"] = [128]  # IMPORTANT, int or None.
    keys["resize_mask_opt_mask"] = [True]


    keys["smooth_img"] = [True]
    keys["smooth_img_ksz"] = [3]
    keys["smooth_img_sigma"] = [.9]

    keys["enhance_color"] = [True]
    keys["enhance_color_fact"] = [3.]

    llists = [keys[k] for k in keys.keys()]
    namekeys = [k for k in keys.keys()]
    cmds = get_combos(llists, namekeys)

    return cmds


def get_hybridloss():
    """
    Get the command line by changing the hyper-parameters of:
    1. loss: hybrid loss.
    :return: list of commands.
    """
    keys = dict()
    keys["loss"] = [constants.HYBRIDLOSS]

    llists = [keys[k] for k in keys.keys()]
    namekeys = [k for k in keys.keys()]
    cmds = get_combos(llists, namekeys)

    return cmds


def get_dataset(dataset_name):
    """
    Get the command line by changing the hyper-parameters of:
    1. dataset
    :return: list of commands.
    """
    keys = dict()
    # constants.GLAS
    # constants.CUB
    # constants.OXF

    keys["dataset"] = [dataset_name]

    announce_msg("Generate configs for {} dataset".format(keys["dataset"]))

    assert len(keys["dataset"]) == 1, "We work with only one dataset." \
                                      "....[NOT OK]"


    if keys['dataset'][0] == constants.GLAS:
        t = 67.
        keys['p_init_samples'] = [100 * 8 / t]  # 5 examples per class. total
        # train 67. with 29 benign, and 38 malignant. (split 0, fold 0). this
        # will give the total selected samples 10 --> 4 per class.

        keys['p_samples'] = [100 * 2 / t]  # this will add 2 samples in
        # total each round --> 1 sample per class.

        keys['max_al_its'] = [25]  # todo. compute it.

    if keys['dataset'][0] == constants.CUB:
        t = 4794.
        keys['p_init_samples'] = [100 * 200 * 1 / t]  # 1 examples per class.
        # total train 4794. (split 0, fold 0). this
        # will give the total selected samples 200 * 1 --> 1 per class.

        keys['p_samples'] = [100 * 200 / t]  # this will add 200 samples in
        # total each round --> 1 sample per class.

        keys['max_al_its'] = [20]  # todo. compute it.

    if keys['dataset'][0] == constants.OXF:
        t = 1020.
        keys['p_init_samples'] = [100 * 102 * 1 / t]  # 1 examples per class.
        # total train 1020.  (split 0, fold 0). this
        # will give the total selected samples 102 * 1 --> 1 per class.

        keys['p_samples'] = [100 * 102 / t]  # this will add 102 samples in
        # total each round --> 1 sample per class.

        keys['max_al_its'] = [9]  # todo. compute it.

    if keys['dataset'][0] == constants.CAM16:
        t = 12174.  # trainset: tumor.
        keys['p_init_samples'] = [100 * 30 * 1 / t]  # 1 examples per class.
        # total train 12174.  (split 0, fold 0). this
        # will give the total selected samples 4 * 1 --> 4 per class.

        keys['p_samples'] = [100 * 1 / t]  # this will add 1 samples in
        # total each round --> 1 sample per class.

        keys['max_al_its'] = [30]  # todo. compute it.

    if keys['dataset'][0] in [constants.GLAS,
                              constants.CUB,
                              constants.OXF,
                              constants.CAM16
                              ]:  # segmentation
        keys['task'] = [constants.SEG]
    else:
        raise ValueError(
            'Dataset {} with unknown task.'.format(keys['datasets'])
        )

    # for everyone
    keys["split"] = [0]
    keys["fold"] = [0]

    llists = [keys[k] for k in keys.keys()]
    namekeys = [k for k in keys.keys()]

    return get_combos(llists, namekeys)


def get_all_configs(do_epochs,
                    al_type=constants.AL_FULL_SUP,
                    dataset_name=constants.CUB
                    ):
    """
    Get the command line by changing the hyper-parameters of many things.
    :return: list of commands where each one represents of configuration to
    be run.
    """
    dataset_params = get_dataset(dataset_name=dataset_name)

    al_params = get_al(al_type)
    subtask = get_val(al_params[0], "subtask")
    epochs = None
    if do_epochs is not None:
        epochs = do_epochs[dataset_name]
    llists = [
        get_opt_model(dataset=dataset_name,
                      epochs=epochs,
                      subtask=subtask,
                      al_type=al_type
                      ),
        dataset_params,
        al_params,
        get_optmi_mask()
    ]
    configs = get_combos(llists, None)
    return configs


def get_val(config_cmd, att):
    """
    Get the value of an attribute in a configuration command.
    :param config_cmd: str. configuration command.
    :param att: str. attribute.
    :return:
    """
    words = config_cmd.split(" ")
    i = words.index("--{}".format(att))
    val = words[i + 1]

    return val


def get_configs(dataset, al_type):
    """
    Returns the configs.
    :return:
    """
    keys = dict()

    if dataset == constants.GLAS:
        keys['knn'] = [40]

        if al_type == constants.AL_WSL:
            keys["freeze_classifier"] = [False]
        else:
            keys["freeze_classifier"] = [True]

        # segloss
        keys['segloss_l'] = [constants.BinCrossEntropySegmLoss]
        keys['segloss_pl'] = [constants.BinCrossEntropySegmLoss]

        keys['scale_cl'] = [1.]
        keys['scale_seg'] = [1.]
        keys['scale_seg_u'] = [0.1]  # IMPORTANT
        keys['scale_seg_u_sch'] = [constants.ConstantWeight]
        keys['scale_seg_u_end'] = [1e-9]
        keys['scale_seg_u_sigma'] = [150.]
        keys['scale_seg_u_p_abstention'] = [0.0]


        keys["weight_pseudo_loss"] = [False]

        keys["protect_pl"] = [True]

        keys["estimate_pseg_thres"] = [True]

        if al_type == constants.AL_WSL:
            keys["estimate_seg_thres"] = [False]  # use 0.5
        else:
            keys["estimate_seg_thres"] = [True]

        keys["seg_threshold"] = [0.5]


        # model Un-Net
        keys["base_width"] = [24]
        keys["leak"] = [64]

        # total optimization: pseudo-labeled samples.
        keys['seg_elbon'] = [False]


    if dataset == constants.CAM16:
        keys['knn'] = [40]

        if al_type == constants.AL_WSL:
            keys["freeze_classifier"] = [False]
        else:
            keys["freeze_classifier"] = [True]

        # segloss
        keys['segloss_l'] = [constants.BinCrossEntropySegmLoss]
        keys['segloss_pl'] = [constants.BinCrossEntropySegmLoss]

        keys['scale_cl'] = [1.]
        keys['scale_seg'] = [1.]
        keys['scale_seg_u'] = [0.0001]
        keys['scale_seg_u_sch'] = [constants.ConstantWeight]
        keys['scale_seg_u_end'] = [1e-9]
        keys['scale_seg_u_sigma'] = [150.]
        keys['scale_seg_u_p_abstention'] = [0.0]


        keys["weight_pseudo_loss"] = [False]

        keys["protect_pl"] = [True]

        keys["estimate_pseg_thres"] = [True]

        if al_type == constants.AL_WSL:
            keys["estimate_seg_thres"] = [False]  # use 0.5
        else:
            keys["estimate_seg_thres"] = [True]

        keys["seg_threshold"] = [0.5]


        # model Un-Net
        keys["base_width"] = [12]
        keys["leak"] = [32]

        # total optimization: pseudo-labeled samples.
        keys['seg_elbon'] = [False]


    if dataset == constants.CUB:
        keys['knn'] = [40]

        if al_type == constants.AL_WSL:
            keys["freeze_classifier"] = [False]
        else:
            keys["freeze_classifier"] = [True]

        # segloss
        keys['segloss_l'] = [constants.BinCrossEntropySegmLoss]
        keys['segloss_pl'] = [constants.BinCrossEntropySegmLoss]

        keys['scale_cl'] = [1.]
        keys['scale_seg'] = [1.]
        keys['scale_seg_u'] = [0.001]
        keys['scale_seg_u_sch'] = [constants.ConstantWeight]
        keys['scale_seg_u_end'] = [1e-9]
        keys['scale_seg_u_sigma'] = [150.]
        keys['scale_seg_u_p_abstention'] = [0.0]


        keys["weight_pseudo_loss"] = [False]

        keys["protect_pl"] = [True]

        keys["estimate_pseg_thres"] = [True]

        if al_type == constants.AL_WSL:
            keys["estimate_seg_thres"] = [False]  # use 0.5
        else:
            keys["estimate_seg_thres"] = [True]

        keys["seg_threshold"] = [0.5]


        # model Un-Net
        keys["base_width"] = [12]
        keys["leak"] = [32]

        # total optimization: pseudo-labeled samples.
        keys['seg_elbon'] = [False]

    if dataset == constants.OXF:
        keys['knn'] = [40]

        if al_type == constants.AL_WSL:
            keys["freeze_classifier"] = [False]
        else:
            keys["freeze_classifier"] = [True]

        # segloss
        keys['segloss_l'] = [constants.BinCrossEntropySegmLoss]
        keys['segloss_pl'] = [constants.BinCrossEntropySegmLoss]

        keys['scale_cl'] = [1.]
        keys['scale_seg'] = [1.]
        keys['scale_seg_u'] = [0.1]
        keys['scale_seg_u_sch'] = [constants.ConstantWeight]
        keys['scale_seg_u_end'] = [1e-9]
        keys['scale_seg_u_sigma'] = [150.]
        keys['scale_seg_u_p_abstention'] = [0.0]


        keys["weight_pseudo_loss"] = [False]

        keys["protect_pl"] = [True]

        keys["estimate_pseg_thres"] = [True]

        if al_type == constants.AL_WSL:
            keys["estimate_seg_thres"] = [False]  # use 0.5
        else:
            keys["estimate_seg_thres"] = [True]

        keys["seg_threshold"] = [0.5]


        # model Un-Net
        keys["base_width"] = [24]
        keys["leak"] = [64]

        # total optimization: pseudo-labeled samples.
        keys['seg_elbon'] = [False]

    llists = [keys[k] for k in keys.keys()]
    namekeys = [k for k in keys.keys()]

    return get_combos(llists, namekeys)


def create_jobs(dataset, al_type):
    """
    Create jobs.
    :return:
    """

    # LP
    backbone_dropout = {
        constants.GLAS: 0.2,
        constants.CUB: 0.2,
        constants.OXF: 0.2,
        constants.CAM16: 0.2
    }
    mcdropout_t = {
        constants.GLAS: 50,
        constants.CUB: 50,
        constants.OXF: 50,
        constants.CAM16: 50
    }

    # ==========================================================================
    # choose type of active learning. see constants.py
    # ==========================================================================
    al_types = [al_type]
    # ==========================================================================

    do_epochs = None


    config = get_all_configs(do_epochs=do_epochs,
                             al_type=al_types[0],
                             dataset_name=dataset
                             )

    max_al_rounds = int(get_val(config[0], "max_al_its"))
    MAX_REPEAT = 5

    if al_types[0] in [constants.AL_FULL_SUP, constants.AL_WSL]:
        max_al_rounds = 1

    current_dataset = dataset

    # file contains all the commands.
    run_file = "run-{}-{}.sh".format(current_dataset, al_types[0])
    frun = open(run_file, "w")
    frun.write("#!/usr/bin/env bash \n")

    # backbone_dropout for mcdropout.
    kkeys = ['backbone_dropout', 'mcdropout_t']
    if al_types[0] == constants.AL_MCDROPOUT:
        vals = [[backbone_dropout[current_dataset]],
                [mcdropout_t[current_dataset]]
                ]
    else:
        vals = [[0.0], [0]]
    config = get_combos([config, get_combos(vals, kkeys)], None)

    assert len(config) == 1, 'nbr configs must be 1. found {}'.format(
        len(config))

    subtask = get_val(config[0], "subtask")
    cnd = (subtask == constants.SUBCLSEG)
    cnd |= ((subtask == constants.SUBCL) and (al_types[0] == constants.AL_WSL))

    if cnd:
        subtask = ""
    elif subtask == constants.SUBCL:
        subtask = "-{}".format(subtask)

    debug_folders = [
        'paper_label_prop{}/{}/{}'.format(
            subtask,
            current_dataset,
            al_types[0]
        )]


    kkeys = ['debug_subfolder']
    vals = [debug_folders]
    config = get_combos([config, get_combos(vals, kkeys)], None)

    extra_conf = get_configs(dataset=current_dataset,
                             al_type=al_types[0]
                             )

    lconfigs = get_combos([config, extra_conf], None)
    # ==========================================================================

    jb_counter = 0
    myseeds = list(range(0, MAX_REPEAT, 1))

    cnt_sbtatch = 0  # how many jobs we submit.
    for zz, myseed in enumerate(myseeds):
        frun.write('\n# ' + '=' * 78 + '\n')
        if zz == 0:
            frun.write('cudaid=$1\n')

        frun.write('\n#  Start MYSEED: {} \n'.format(myseed))

        # generate per-seed run file. helpful on our servers but not cc.
        seed_run_file = "run-{}-{}-myseed-{}.sh".format(
            current_dataset,
            al_types[0],
            myseed
        )
        # per seed file
        sfrun = open(seed_run_file, "w")
        sfrun.write("#!/usr/bin/env bash \n")
        sfrun.write('\n# ' + '=' * 78 + '\n')
        sfrun.write('cudaid=$1\n')
        sfrun.write('\n#  Start MYSEED: {} \n'.format(myseed))

        al_type = al_types[0]

        loss = None
        current_task = get_val(lconfigs[0], 'task')
        if current_task == constants.CL:
            raise ValueError('ERROR. Unsupported. LEAVING.')
        elif current_task == constants.SEG:
                loss = get_hybridloss()
        else:
            raise ValueError("Unknown task {}".format(current_task))

        for n_conf, config in enumerate(lconfigs):
            active_learning = [al_type]
            kal = ['al_type']

            config = get_combos([
                [config], loss, get_combos([active_learning], kal)], None)

            exp_id = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
            exp_id = '{}__{}'.format(
                exp_id, np.random.randint(low=0, high=10000000, size=1)[0])
            # ==============================================================
            #                      START: ACTIVE LEARNING ROUNDS
            # ==============================================================
            ranger = range(max_al_rounds)

            for rnd in ranger:
                vals = [[rnd], [exp_id]]
                knames = ['al_it', 'exp_id']

                # final command
                config_rnd = get_combos([
                    config, get_combos(vals, knames)], None)[0]
                print("Active learning method `{}`,  "
                      "round {}/ max {}.".format(
                        al_type, rnd, max_al_rounds
                        ))
                # check:
                if al_type == constants.AL_WSL:
                    cnd = (get_val(config_rnd, "subtask") ==
                           constants.SUBCL)
                    assert cnd, "ERROR"

                p_pret = get_val(config_rnd, 'path_pre_trained')
                if get_val(config_rnd, "subtask") == constants.SUBCL:
                    assert p_pret == 'None', 'ERROR'


                runthis = "time python main.py " \
                          "--yaml {}.yaml --MYSEED {} {} \n \n".format(
                            get_val(config_rnd, "dataset"),
                            myseed,
                            config_rnd
                            )
                runthis = wrap_command_line(runthis)
                runthis = "\n\n\n" + runthis

                # Write into run.sh for LIVIA.
                frun.write(runthis)
                frun.write('\n# ' + '=' * 78 + '\n')

                sfrun.write(runthis)
                sfrun.write('\n# ' + '=' * 78 + '\n')



                # =============== into the job file.
                cnt_sbtatch += 1
            # ==============================================================
            #                      END: ACTIVE LEARNING ROUNDS
            # ==============================================================


        jb_counter += 1

        sfrun.close()
        os.system("chmod +x {}".format(seed_run_file))

    frun.close()
    os.system("chmod +x {}".format(run_file))

    print("Total number of experiments: {}.".format(cnt_sbtatch))


def wrap_command_line(cmd):
    """
    Wrap command line
    :param cmd: str. command line with space as a separator.
    :return:
    """
    return " \\\n".join(textwrap.wrap(
        cmd, width=77, break_long_words=False, break_on_hyphens=False))


if __name__ == "__main__":
    # list of all the supported active learning types.
    l_al_types = [constants.AL_WSL,
                  constants.AL_RANDOM,
                  constants.AL_LP,
                  constants.AL_FULL_SUP,
                  constants.AL_ENTROPY,
                  constants.AL_MCDROPOUT
                  ]
    # list of all the supported datasets
    l_datasets = [constants.GLAS,
                  constants.CUB
                  ]

    # generate all the bash scripts to run the exps.
    for ds in l_datasets:
        for al_t in l_al_types:
            create_jobs(dataset=ds, al_type=al_t)
