# Sel-contained-as-possible module handles parsing the input using argparse.
# handles seed, and initializes some modules for reproducibility.

import os
from os.path import join
import sys
import argparse
from copy import deepcopy
import warnings


import yaml

import constants

import reproducibility


def str2bool(v):
    """
    Read `v`: and returns a boolean value:
    True: if `v== "True"`
    False: if `v=="False"`
    :param v: str.
    :return: bool.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v == "True":
            return True
        elif v == "False":
            return False
        else:
            raise ValueError(
                "Expected value: 'True'/'False'. found {}.".format(v))
    else:
        raise argparse.ArgumentTypeError('String boolean value expected: '
                                         '"True"/"Flse"')



class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_yaml_args(input_args):
    """
    Gets the yaml arguments.
    Adds the provided seed to `os.environ` using the kee `MYSEED`.

    :param input_args: the output of parser.parse_args().
    :return:
        args: object, where each attribute is an element from parsing
        the yaml file.
        args_dict: dicts, the same as the previous one, but a dictionary.
         (see code).
    """
    # list of allowed variables to be override.
    parser = argparse.ArgumentParser()
    with open(join("./config_yaml/", input_args.yaml), 'r') as f:
        args = yaml.load(f)

        args["yaml"] = input_args.yaml  # mandatory.

        # Checking
        if args["dataset"] == constants.CIFAR_10:
            msg = "'cifar-10' has 10 classes. Found {}.".format(
                args['num_classes'])
            assert args["num_classes"] == 10, msg
        elif args['dataset'] == constants.CIFAR_100:
            msg = "'cifar-100' has 100 classes. Found {}.".format(
                args['num_classes'])
            assert args["num_classes"] == 100, msg
        elif args["dataset"] == constants.SVHN:
            msg = "'svhn' has 10 classes. Found {}.".format(
                args['num_classes'])
            assert args["num_classes"] == 10, msg
        elif args["dataset"] == constants.GLAS:
            msg = "'glas' has 2 classes. Found {}.".format(
                args['num_classes'])
            assert args["num_classes"] == 2, msg

        # Allow the user to override some values in the yaml.
        # This helps changing the hyper-parameters using the command line
        # without changing the yaml file (very helpful during debug!!!!).
        # Create a new parser.

        parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
        parser.add_argument("--yaml", type=str,
                            help="yaml configuration basefile.")
        parser.add_argument("--MYSEED", type=str, default=None, help="Seed.")

        parser.add_argument(
            "--debug_subfolder", type=str, default=None,
            help="Name of subfolder that is used for debugging. Default: ''.")

        parser.add_argument(
            "--dataset", type=str, default=None,
            help="Name of the dataset.")
        parser.add_argument(
            "--num_classes", type=int, default=None,
            help="Number of classes in the dataset.")
        parser.add_argument(
            "--num_masks", type=int, default=None,
            help="Number of masks to produce. For now only one is supported.")

        parser.add_argument(
            "--crop_size", type=int, default=None,
            help="Crop size (int) of the patches during training.")

        parser.add_argument(
            "--split", type=int, default=None,
            help="Split number.")
        parser.add_argument(
            "--fold", type=int, default=None,
            help="Fold number.")
        parser.add_argument(
            "--fold_folder", type=str, default=None,
            help="Relative path to the folder of the folds.")
        parser.add_argument(
            "--up_scale_small_dim_to", type=int, default=None,
            help="The int size to which upscale the small dimension to.")
        parser.add_argument(
            "--resize_h_to", type=int, default=None,
            help="The int size to which rescale the height of the original "
                 "images.")
        parser.add_argument(
            "--resize_mask", type=str2bool, default=None,
            help="Whether to resize the original mask or not.")
        parser.add_argument(
            "--resize_h_to_opt_mask", type=int, default=None,
            help="The int size to which rescale the height of the original "
                 "images in the case of optimizing a mask.")
        parser.add_argument(
            "--resize_mask_opt_mask", type=str2bool, default=None,
            help="Whether to resize the original mask or not in the case of "
            "optimizing a mask.")
        parser.add_argument(
            "--ratio_scale_patch", type=float, default=None,
            help="Rate to which to scale the cropped patches to (float).")
        parser.add_argument(
            "--padding_ratio", type=float, default=None,
            help="The padding ratio for the image (top/bottom) and ("
                 "left/right).")
        parser.add_argument(
            "--pad_eval", type=str2bool, default=None,
            help="Whether to pad the samples during evaluation time.")
        parser.add_argument(
            "--scale_algo", type=int, default=None,
            help="Type of the rescaling algorithm. See PIL.Image.resize.")
        parser.add_argument(
            "--max_epochs", type=int, default=None, help="Max epoch.")
        parser.add_argument(
            "--batch_size", type=int, default=None,
            help="Training batch size (optimizer).")
        parser.add_argument(
            "--valid_batch_size", type=int, default=None,
            help="Validation batch size (optimizer).")
        parser.add_argument(
            "--num_workers", type=int, default=None,
            help="Number of workers for dataloader multi-processing.")
        parser.add_argument(
            "--height_tag", type=int, default=None,
            help="Height of the tags for the figure of the segmentation.")

        # ======================================================================
        #                      OPTIMIZER
        # ======================================================================
        # opt0: optimizer for the model.
        parser.add_argument(
            "--optn0__name_optimizer", type=str, default=None,
            help="Name of the optimizer 'sgd', 'adam'.")
        parser.add_argument(
            "--optn0__lr", type=float, default=None,
            help="Learning rate (optimizer)")
        parser.add_argument(
            "--optn0__momentum", type=float, default=None,
            help="Momentum (optimizer)")
        parser.add_argument(
            "--optn0__dampening", type=float, default=None,
            help="Dampening for Momentum (optimizer)")
        parser.add_argument(
            "--optn0__nesterov", type=str2bool, default=None,
            help="Nesterov or not for Momentum (optimizer)")
        parser.add_argument(
            "--optn0__weight_decay", type=float, default=None,
            help="Weight decay (optimizer)")
        parser.add_argument(
            "--optn0__beta1", type=float, default=None,
            help="Beta1 for adam (optimizer)")
        parser.add_argument(
            "--optn0__beta2", type=float, default=None,
            help="Beta2 for adam (optimizer)")
        parser.add_argument(
            "--optn0__eps_adam", type=float, default=None,
            help="eps for adam (optimizer)")
        parser.add_argument(
            "--optn0__amsgrad", type=str2bool, default=None,
            help="amsgrad for adam (optimizer)")
        parser.add_argument(
            "--optn0__lr_scheduler", type=str2bool, default=None,
            help="Whether to use or not a lr scheduler")
        parser.add_argument(
            "--optn0__name_lr_scheduler", type=str, default=None,
            help="Name of the lr scheduler")
        parser.add_argument(
            "--optn0__gamma", type=float, default=None,
            help="Gamma of the lr scheduler. (mystep)")
        parser.add_argument(
            "--optn0__last_epoch", type=int, default=None,
            help="The index of the last epoch where to stop adjusting the LR."
                 " (mystep)")
        parser.add_argument(
            "--optn0__min_lr", type=float, default=None,
            help="Minimum allowed value for lr. (mystep lr scheduler)")
        parser.add_argument(
            "--optn0__t_max", type=float, default=None,
            help="T_max, maximum epochs to restart. (cosine lr scheduler)")
        parser.add_argument(
            "--optn0__step_size", type=int, default=None,
            help="Step size for lr scheduler.")

        # ======================================================================
        #                              MODEL
        # ======================================================================
        parser.add_argument(
            "--alpha", type=float, default=None,
            help="Alpha (classifier, wildcat)")
        parser.add_argument(
            "--kmax", type=float, default=None,
            help="Kmax (classifier, wildcat)")
        parser.add_argument(
            "--kmin", type=float, default=None,
            help="Kmin (classifier, wildcat)")
        parser.add_argument(
            "--dropout", type=float, default=None,
            help="Dropout (classifier, wildcat)")
        parser.add_argument(
            "--modalities", type=int, default=None,
            help="Number of modalities (classifier, wildcat)")
        parser.add_argument(
            "--name_model", type=str, default=None,
            help="Name of the model: `lenet`, ...")
        parser.add_argument(
            "--base_width", type=int, default=None,
            help="Base width of the upscale-part of U-Net.")
        parser.add_argument(
            "--leak", type=int, default=None,
            help="Number of the feature maps to be extracted from the "
                 "classifier feature maps to be used for segmentation.")
        parser.add_argument(
            "--backbone", type=str, default=None,
            help="Name of the backbone (for SEG task).")
        parser.add_argument(
            "--output_stride", type=int, default=None,
            help="Output stride for Deeplabv3+ head. supported values [8, 16].")
        parser.add_argument(
            "--freeze_bn", type=str2bool, default=None,
            help="Whether to freeze or not the bn parameters of deeplab model.")
        parser.add_argument(
            "--path_pre_trained", type=str, default=None,
            help="Absolute/relative path to file containing parameters of a "
                 "model. Use --strict to specify if the  pre-trained "
                 "model needs to match exactly the current model or not.")
        parser.add_argument(
            "--strict", type=str2bool, default=None,
            help="If True, the pre-trained model needs to match exactly the "
                 "current model. Default: True.")
        parser.add_argument(
            "--pre_trained", type=str2bool, default=None,
            help="If True, load pre-trained model. Default: False.")
        parser.add_argument(
            "--dropoutnetssl", type=float, default=None,
            help="Dropout (classifier SOTASSL)")
        parser.add_argument(
            "--backbone_dropout", type=float, default=None,
            help="Dropout for the backbone. useful for Bayesian active "
                 "learning and other techs.")

        # ======================================================================
        #                         ACTIVE LEARNING
        # ======================================================================
        parser.add_argument(
            "--al_it", type=int, default=None,
            help="Active learning round number.")
        parser.add_argument(
            "--max_al_its", type=int, default=None,
            help="Active learning  maximum rounds.")
        parser.add_argument(
            "--al_type", type=str, default=None,
            help="Active learning selection type. see constants.py")
        parser.add_argument(
            "--p_init_samples", type=float, default=None,
            help="Percentage of init. samples to start active learning rounds.")
        parser.add_argument(
            "--p_samples", type=float, default=None,
            help="Percentage of samples to select at each round.")
        parser.add_argument(
            "--task", type=str, default=None,
            help="Task: see constants.tasks.")
        parser.add_argument(
            "--subtask", type=str, default=None,
            help="Subtask: see constants.subtasks.")
        parser.add_argument(
            "--exp_id", type=str, default=None,
            help="Exp. ID.")
        parser.add_argument(
            "--loss", type=str, default=None,
            help="Training loss.")
        parser.add_argument(
            "--segloss_l", type=str, default=None,
            help="Segmentation training loss for labeled samples. (SEG task)")
        parser.add_argument(
            "--segloss_pl", type=str, default=None,
            help="Segmentation training loss for pseudo-labeled samples. (SEG "
            "task)")
        parser.add_argument(
            "--scale_cl", type=float, default=None,
            help="How much to scale the classification loss.")
        parser.add_argument(
            "--scale_seg", type=float, default=None,
            help="How much to scale the supervised segmentation loss.")
        parser.add_argument(
            "--scale_seg_u", type=float, default=None,
            help="How much to scale the unsupervised segmentation loss.")
        parser.add_argument(
            "--scale_seg_u_end", type=float, default=None,
            help="Lower allowed value for scaling the unsupervised "
                 "segmentation loss.")
        parser.add_argument(
            "--scale_seg_u_sigma", type=float, default=None,
            help="Sigma for the exp schedule for scaling the unsupervised "
                 "segmentation loss.")
        parser.add_argument(
            "--scale_seg_u_p_abstention", type=float, default=None,
            help="Probability of a pixel-pseudo-annotation to be ignored.")
        parser.add_argument(
            "--weight_pseudo_loss", type=str2bool, default=None,
            help="If true, the loss of pseudo-labeled samples is weighted "
                 "using 1./nbr_pseudo_labeled_samples.")
        parser.add_argument(
            "--scale_seg_u_sch", type=str, default=None,
            help="Decay schedule for scaling the unsupervised "
                 "segmentation loss.")
        parser.add_argument(
            "--seg_smooth", type=float, default=None,
            help="smooth value Dice loss. (SEG task)")
        parser.add_argument(
            "--seg_elbon", type=str2bool, default=None,
            help="Whether to use ELB to optimize pseudo-labeled samples. if "
                 "it is not used, such samples are considered as fully "
                 "supervised.")
        parser.add_argument(
            "--seg_init_t", type=float, default=None,
            help="init_t for ELB for pseudo-segmented samples.")
        parser.add_argument(
            "--seg_max_t", type=float, default=None,
            help="max_t for ELB for pseudo-segmented samples.")
        parser.add_argument(
            "--seg_mulcoef", type=float, default=None,
            help="mulcoef for ELB for pseudo-segmented samples.")
        parser.add_argument(
            "--freeze_classifier", type=str2bool, default=None,
            help="whether to freeze or the the classifier.")
        parser.add_argument(
            "--estimate_pseg_thres", type=str2bool, default=None,
            help="whether to estimate the segmentation threshold from the "
                 "validation set or not for the pseudo-labeled samples. if "
                 "not, a default threshold of 0.5 "
                 "is used.")
        parser.add_argument(
            "--estimate_seg_thres", type=str2bool, default=None,
            help="whether to estimate the segmentation threshold from the "
                 "validation set or not for the prediction. if not, a default "
                 "threshold of 0.5 is used.")
        parser.add_argument(
            "--seg_threshold", type=float, default=None,
            help="Segmentation threshold.")
        parser.add_argument(
            "--clustering", type=str, default=None,
            help="How to select samples in our method (clustering). see "
                 "constants.ours_clustering")

        parser.add_argument(
            "--mcdropout_t", type=int, default=None,
            help="Number of times to sample for AL MC-dropout.")
        parser.add_argument(
            "--knn", type=int, default=None,
            help="Number of neighbors to look for labeled samples around an "
                 "unlabeled sample.")


        parser.add_argument(
            "--pair_w_batch_size", type=int, default=None,
            help="batch size to compute the pair-wise similarities.")

        # histograms
        parser.add_argument(
            "--use_dist_global_hist", type=str2bool, default=None,
            help="Whether to use or not the image global histogram to compute "
                 "the proximity between samples.")
        parser.add_argument(
            "--nbr_bins_histc", type=int, default=None,
            help="Number of bins in histogram.")
        parser.add_argument(
            "--min_histc", type=float, default=None,
            help="Min value in histogram.")
        parser.add_argument(
            "--max_histc", type=float, default=None,
            help="Max value in histogram.")
        parser.add_argument(
            "--sigma_histc", type=float, default=None,
            help="Sigma for histogram.")

        # ======================================================================
        #                                  PROPERTIES
        # ======================================================================
        parser.add_argument(
            "--protect_pl", type=str2bool, default=None,
            help="whether to sample from pseudo-labeled or not when there "
                 "are still unlabeled sample available. ")

        parser.add_argument(
            "--smooth_img", type=str2bool, default=None,
            help="If true, the image is smoothed using a gaussian filter ("
                 "optimize mask).")
        parser.add_argument(
            "--smooth_img_ksz", type=int, default=None,
            help="Size of the gaussian filter used to smooth the image ("
                 "optimize mask).")
        parser.add_argument(
            "--smooth_img_sigma", type=float, default=None,
            help="Sigma of the gaussian filter used to smooth the image ("
                 "optimize mask).")
        parser.add_argument(
            "--enhance_color", type=str2bool, default=None,
            help="If true, color image is enhanced before optimizing the mask ("
                 "optimize mask).")
        parser.add_argument(
            "--enhance_color_fact", type=float, default=None,
            help="color enhancement factor. float [0., [. color enhancing "
                 "factor. 0. gives black and white. 1 gives the same image. "
                 "low values than 1 means less color, higher values than 1 "
                 "means more colors.(for the dataloader) (optimize mask)")

        # ======================================================================
        #                PROPERTIES/MASK OPTIMIZATION:
        #                           STORING
        # ======================================================================
        parser.add_argument(
            "--share_masks", type=str2bool, default=None,
            help="if this is true, the mask obtained from the pair idu_idl is "
                 " stored in a folder that is shared among all the active "
                 " learning rounds to avoid recomputing it again. this saves "
                 "time.")

        # TODO: finish this overriding!
        input_parser = parser.parse_args()

        def warnit(name, vl_old, vl):
            """
            Warn that the variable with the name 'name' has changed its value
            from 'vl_old' to 'vl' through command line.
            :param name: str, name of the variable.
            :param vl_old: old value.
            :param vl: new value.
            :return:
            """
            if vl_old != vl:
                print("Changing {}: {}  -----> {}".format(name, vl_old, vl))
            else:
                print("{}: {}".format(name, vl_old))

        attributes = input_parser.__dict__.keys()

        for k in attributes:
            val_k = getattr(input_parser, k)
            if k in args.keys():
                if val_k is not None:
                    warnit(k, args[k], val_k)
                    args[k] = val_k
                else:
                    warnit(k, args[k], args[k])

            elif k in args['model'].keys():  # try model
                if val_k is not None:
                    warnit('model.{}'.format(k), args['model'][k], val_k)
                    args['model'][k] = val_k
                else:
                    warnit('model.{}'.format(k), args['model'][k],
                           args['model'][k])

            elif k in args['optimizer'].keys():  # try optimizer 0
                if val_k is not None:
                    warnit(
                        'optimizer.{}'.format(k), args['optimizer'][k], val_k)
                    args['optimizer'][k] = val_k
                else:
                    warnit(
                        'optimizer.{}'.format(k), args['optimizer'][k],
                        args['optimizer'][k]
                    )
            else:
                raise ValueError("Key {} was not found in args. ..."
                                 "[NOT OK]".format(k))

        # add the current seed to the os env. vars. to be shared across this
        # process.
        # this seed is expected to be local for this process and all its
        # children.
        # running a parallel process will not have access to this copy not
        # modify it. Also, this variable will not appear in the system list
        # of variables. This is the expected behavior.
        # TODO: change this way of sharing the seed through os.environ. [future]
        # the doc mentions that the above depends on `putenv()` of the
        # platform.
        # https://docs.python.org/3.7/library/os.html#os.environ
        os.environ['MYSEED'] = args["MYSEED"]

        args_dict = deepcopy(args)



        args = Dict2Obj(args)
        # sanity check +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if args.dataset in constants.CL_DATASETS:
            msg = "dataset {} is for task CL but the" \
                  " specified task is {}".format(args.dataset, args.task)
            assert args.task == constants.CL, msg

        if args.dataset in constants.SEG_DATASETS:
            msg = "dataset {} is for task SEG but the" \
                  " specified task is {}".format(args.dataset, args.task)
            assert args.task == constants.SEG, msg

        assert any([args.use_dist_global_hist]), "one must be true."


        if isinstance(args.num_masks, int):
            msg = "'num_masks' must be one. found {}.".format(args.num_masks)
            assert args.num_masks == 1, msg

        if args.freeze_classifier and (args.subtask == constants.SUBCL):
            msg = "'args.freeze_classifier' is True. " \
                  " but 'args.subtask' is {}. we do not expect this " \
                  "configuration simply because there will be no learning. " \
                  "Unless you modified things, this is not a valid config." \
                  "Going to exit." \
                  "".format(constants.SUBCL)
            warnings.warn(msg)
            sys.exit(0)

        if args.al_type == constants.AL_LP:
            msg = "you have to share the pseudo-masks. So, please set " \
                  "`share_masks` to true, because it is false."
            assert args.share_masks, msg

            msg = "'use_dist_global_hist' must be true."
            assert args.use_dist_global_hist, msg

        if args.al_type == constants.AL_WSL:
            msg = "Can't estimate threshold over AL_WSL." \
                  " masks are already binary."
            assert not args.estimate_seg_thres, msg

    return args, args_dict


def parse_input():
    """
    Parse the input.
    and
    initialize some modules for reproducibility.
    """
    parser = argparse.ArgumentParser()
    # mandatory: --yaml basefilename
    parser.add_argument("--yaml", type=str, help="yaml configuration basefile.")
    input_args, _ = parser.parse_known_args()
    args, args_dict = get_yaml_args(input_args)

    # Reproducibility control: this will use the seed in `os.environ['MYSEED']`.

    reproducibility.init_seed()
    # todo: remove input_args. useless. can be accessed from args.
    return args, args_dict, input_args