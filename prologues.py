import os
import sys
from os.path import join, dirname, expanduser, abspath
import subprocess
from copy import deepcopy
import datetime as dt
import shutil
import random
import pickle as pkl
import warnings

import yaml
import torch
from torch.utils.data import DataLoader

from reproducibility import set_default_seed

from deeplearning.pairwise_similarity import PairwiseSimilarity
from deeplearning.sampling import PairSamples


from shared import csv_writer, csv_loader
from shared import wrap_command_line
from shared import announce_msg
from shared import find_files_pattern
from shared import check_if_allow_multgpu_mode

from tools import get_exp_name
from tools import create_folders_for_exp
from tools import copy_code
from tools import log
from tools import plot_curves_from_dict
from tools import plot_curve
from tools import load_pre_pretrained_model
from tools import get_rootpath_2_dataset


from loader import PhotoDataset
from loader import default_collate
from loader import _init_fn
from loader import MyDataParallel


from instantiators import instantiate_models

import constants


from torchvision import transforms

# ==============================================================================
#                         FOR MAIN.PY
#           CONTAINS FUNCTIONS THAT DO PRELIMINARY TASKS BEFORE
#                     STARTING THE MAIN TASK IN
#                     `MAIN.PY` OR OTHER FILES.
# PRELIMINARY TASKS: PARSING INPUT, GET FILES, CREATE FOLDERS, GET SAMPLES,
# GET DATASETS...
# ==============================================================================

def prologue_init(args):
    """
    Prepare writing folders.
    :return:
    """
    # Write in scratch instead of /project
    output = dict()
    subset_target = 'train'
    prox_histo = args.use_dist_global_hist
    placement_scr, parent, exp_name = None, None, None
    placement_node = None

    if "CC_CLUSTER" in os.environ.keys():
        parent = "exps-active-learning/weak-segmentation"
        if args.debug_subfolder != '':
            parent = join(parent, args.debug_subfolder)
        # placement
        placement_node = '{}'.format(os.environ["SLURM_TMPDIR"])
        # fixed path
        exp_name = get_exp_name(args)

        tag = [('id', args.exp_id),
               ('task', args.task),
               ('subtask', args.subtask),
               ('al', args.al_type),
               ('ds', args.dataset),
               ('seed', args.MYSEED)
               ]
        tag = [(el[0], str(el[1])) for el in tag]
        tag = '-'.join(['_'.join(el) for el in tag])

        if args.al_type == constants.AL_LP:
            tag2 = [('clustr', args.clustering),
                    ('knn', args.knn),
                    ('scale_seg_u', args.scale_seg_u),
                    ('seg_elbon', args.seg_elbon)
                    ]
            tag2 = [(el[0], str(el[1])) for el in tag2]
            tag2 = '-'.join(['_'.join(el) for el in tag2])
            tag = "{}-{}".format(tag, tag2)

        COMMON = join(placement_node, parent, tag, "common")
        VISION = join(placement_node, parent, tag, "vision")
        SHARED_OPT_MASKS = join(placement_node, parent, tag, "shared_opt_masks")
        ROUNDS = join(placement_node, parent, tag)
        OUTD = join(placement_node, parent, tag, exp_name)
        placement_scr = '{}'.format(os.environ["SCRATCH"])

        # where we store sims
        tag_sims = [
            ('dataset', args.dataset),
            ('subset', subset_target),
            ('fold', args.fold),
            ('split', args.split),
            ('usehisto', prox_histo)
        ]
        tag_sims = [(el[0], str(el[1])) for el in tag_sims]
        tag_sims = '-'.join(['_'.join(el) for el in tag_sims])

        SIMS = join(placement_node, parent, "pairwise_sims", tag_sims)
    else:
        # we need to write in home...
        # parent_folder = dirname(abspath(__file__)).split(os.sep)[-1]
        # FOLDER = join("{}/code".format(os.environ["NEWHOME"]), parent_folder)
        # OUTD = join(FOLDER, "exps", get_exp_name(args))

        tag = [('id', args.exp_id),
               ('task', args.task),
               ('subtask', args.subtask),
               ('al', args.al_type),
               ('ds', args.dataset),
               ('seed', args.MYSEED)
               ]
        tag = [(el[0], str(el[1])) for el in tag]
        tag = '-'.join(['_'.join(el) for el in tag])

        if args.al_type == constants.AL_LP:
            tag2 = [('clustr', args.clustering),
                    ('knn', args.knn),
                    ('scale_seg_u', args.scale_seg_u),
                    ('seg_elbon', args.seg_elbon)
                    ]
            tag2 = [(el[0], str(el[1])) for el in tag2]
            tag2 = '-'.join(['_'.join(el) for el in tag2])
            tag = "{}-{}".format(tag, tag2)

        parent_lv = "exps"
        if args.debug_subfolder != '':
            parent_lv = join(parent_lv, args.debug_subfolder)
        OUTD = join(dirname(abspath(__file__)),
                    parent_lv, tag,
                    get_exp_name(args)
                    )
        COMMON = join(dirname(abspath(__file__)),
                      parent_lv,
                      tag,
                      "common"
                      )
        VISION = join(dirname(abspath(__file__)),
                      parent_lv,
                      tag,
                      "vision"
                      )
        SHARED_OPT_MASKS = join(dirname(abspath(__file__)),
                                parent_lv,
                                tag,
                                "shared_opt_masks"
                                )
        ROUNDS = join(dirname(abspath(__file__)),
                      parent_lv,
                      tag
                      )

        OUTD = expanduser(OUTD)
        # where we store sims
        tag_sims = [
            ('dataset', args.dataset),
            ('subset', subset_target),
            ('fold', args.fold),
            ('split', args.split),
            ('usehisto', prox_histo)
        ]
        tag_sims = [(el[0], str(el[1])) for el in tag_sims]
        tag_sims = '-'.join(['_'.join(el) for el in tag_sims])

        SIMS = join(dirname(abspath(__file__)), "pairwise_sims", tag_sims)

    output['COMMON'] = COMMON
    output['OUTD'] = OUTD
    output['VISION'] = VISION
    output['SHARED_OPT_MASKS'] = SHARED_OPT_MASKS
    output['ROUNDS'] = ROUNDS
    output['tag_sims'] = tag_sims
    output['SIMS'] = SIMS
    output['tag'] = tag
    output['placement_scr'] = placement_scr
    output['parent'] = parent
    output['exp_name'] = exp_name
    output['placement_node'] = placement_node

    return output

def check_if_round_already_done(placement_scr,
                                parent,
                                tag,
                                exp_name,
                                OUTD,
                                VISION,
                                COMMON,
                                SHARED_OPT_MASKS
                                ):
    """
    Check if this AL round has already been done or not.
    if yes, exit.

    :param placement_scr:
    :param parent:
    :param tag:
    :param exp_name:
    :param OUTD:
    :return:
    """
    # check if this round has already been processed.

    if "CC_CLUSTER" in os.environ.keys():
        FINAL_WRITE_FD = join(placement_scr, parent, tag, exp_name)
    else:
        FINAL_WRITE_FD = OUTD

    gate = os.path.isfile(join(FINAL_WRITE_FD, "end.txt"))
    gate = gate or os.path.isfile(join(FINAL_WRITE_FD, "best_model.pt"))
    if gate:
        if "CC_CLUSTER" in os.environ.keys():
            # clean shared opt masks (dynamic folder. keep only the last state)
            # this will cause an error since this folder does not exist
            # because each run has a new process id. and since the path in
            # the node contain the process id, this command will raise an error.
            # cmd = "rm -r {}/* ".format(SHARED_OPT_MASKS)
            # print("Running this command: '{}'".format(cmd))
            #
            # subprocess.run(cmd, shell=True, check=True)
            # copy common, vision, shared_top_masks that have been a;ready
            # processed into node
            SCRATCH_FD = join(placement_scr, parent, tag)
            FD_COPY = ["vision", "common", "shared_opt_masks"]
            for fd_sc, fd_node in zip(FD_COPY,
                                      [VISION, COMMON, SHARED_OPT_MASKS]):
                # check if the folder is not empty.
                # cp -r folder/* will raise error `cp: cannot stat` if it is
                # the case.
                if len(os.listdir(join(SCRATCH_FD, fd_sc))) > 0:
                    cmd = "cp -r {}/* {}".format(
                        join(SCRATCH_FD, fd_sc),
                        fd_node
                    )
                    print("Running bash-cmd: \n{}".format(cmd))
                    subprocess.run(cmd, shell=True, check=True)

        sys.exit(0)  # leave since this round has already been processed.
    # ==========================================================================


def prologue_fds_0(args,
                   OUTD,
                   COMMON,
                   SIMS,
                   VISION,
                   ROUNDS,
                   SHARED_OPT_MASKS
                   ):
    """
    Do some stuff before starting.
    :return:
    """
    lfolders = [OUTD, COMMON, SIMS, VISION, ROUNDS, SHARED_OPT_MASKS]

    for fdxx in lfolders:
        if not os.path.exists(fdxx):
            os.makedirs(fdxx)

    OUTD_TR = create_folders_for_exp(OUTD, "train")
    OUTD_VL = create_folders_for_exp(OUTD, "valid")
    OUTD_TS = create_folders_for_exp(OUTD, "test")
    OUTD_TLB = create_folders_for_exp(OUTD, "tlb")
    OUTD_LO = create_folders_for_exp(OUTD, "leftovers")

    OUTD_OPTMASKS = join(OUTD, "optim_masks")  # hosts many sub-folders.
    if not os.path.exists(OUTD_OPTMASKS):
        os.makedirs(OUTD_OPTMASKS)

    fd_p_msks = None
    if args.share_masks:
        msg = "{} is not a valid directory.".format(SHARED_OPT_MASKS)
        assert os.path.isdir(SHARED_OPT_MASKS), msg
        fd_p_msks = SHARED_OPT_MASKS
    else:
        msg = "{} is not a valid directory.".format(OUTD_OPTMASKS)
        fd_p_msks = join(OUTD_OPTMASKS, "bin_masks")  # it has already been
        # created.
        assert os.path.isdir(fd_p_msks), msg

    return {
        'OUTD_TR': OUTD_TR,
        'OUTD_VL': OUTD_VL,
        'OUTD_TS': OUTD_TS,
        'OUTD_TLB': OUTD_TLB,
        'OUTD_LO': OUTD_LO,
        'OUTD_OPTMASKS': OUTD_OPTMASKS,
        'fd_p_msks': fd_p_msks
    }


def prologue_fds_1(args,
                   OUTD,
                   input_args,
                   args_dict
                   ):
    """
    Do some stuff before starting.
    :return:
    """
    subdirs = ["init_params"]
    for sbdr in subdirs:
        if not os.path.exists(join(OUTD, sbdr)):
            os.makedirs(join(OUTD, sbdr))

    # save the yaml file and input config.
    if not os.path.exists(join(OUTD, "code/")):
        os.makedirs(join(OUTD, "code/"))

    with open(join(OUTD, "code/", input_args.yaml), 'w') as fyaml:
        yaml.dump(args_dict, fyaml)
    str_cmd = "time python " + " ".join(sys.argv)
    str_cmd = wrap_command_line(str_cmd)
    with open(join(OUTD, "code/cmd.sh"), 'w') as frun:
        frun.write("#!/usr/bin/env bash \n")
        frun.write(str_cmd)

    copy_code(join(OUTD, "code/"), compress=True, verbose=True)

    training_log = join(OUTD, "training.txt")
    results_log = join(OUTD, "results.txt")

    return training_log, results_log


def get_csv_files(args):
    """
    Get the csv files.
    :return:
    """
    relative_fold_path = join(
        args.fold_folder, args.dataset, "split_" + str(args.split),
                                        "fold_" + str(args.fold)
    )
    if isinstance(args.name_classes, str):  # path
        path_classes = join(relative_fold_path, args.name_classes)
        assert os.path.isfile(path_classes), "File {} does not exist .... " \
                                             "[NOT OK]".format(path_classes)
        with open(path_classes, "r") as fin:
            args.name_classes = yaml.load(fin)
    csvfiles = []
    for subp in ["train_s_", "valid_s_", "test_s_"]:
        csvfiles.append(
            join(
                relative_fold_path, "{}{}_f_{}.csv".format(
                    subp, args.split, args.fold))
        )

    train_csv, valid_csv, test_csv = csvfiles

    # Check if the csv files exist. If not, raise an error.
    if not all([os.path.isfile(fcsv) for fcsv in csvfiles]):
        checkin = [os.path.isfile(fcsv) for fcsv in csvfiles]
        raise ValueError(
            "At least one of the files does not exist *.cvs: \n {} \n "
            "{}".format(csvfiles, checkin)
        )

    return train_csv, valid_csv, test_csv


def compute_similarities(args,
                         tag_sims,
                         train_csv,
                         rootpath,
                         DEVICE,
                         SIMS,
                         training_log,
                         placement_node,
                         parent
                         ):
    """
    Compute similarities.
    :return:
    """
    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.
    cnd_drop_n = (args.dataset == constants.CAM16)
    cnd_drop_n &= (args.al_type != constants.AL_WSL)

    if args.al_type != constants.AL_LP:  # get out.
        return 0

    # 1. compute sims
    current_dir = dirname(abspath(__file__))

    # compute proximity
    if not os.path.exists(
            join("pairwise_sims", '{}.tar.gz'.format(tag_sims))):
        announce_msg("Going to project samples, and compute apirwise "
                     "similarities")

        all_train_samples = csv_loader(train_csv,
                                       rootpath,
                                       drop_normal=cnd_drop_n
                                       )
        for ii, el in enumerate(all_train_samples):
            el[4] = constants.L  # just for the loader consistency.
            # masks are not used when computing the pairwise similarity.

        set_default_seed()
        compute_sim = PairwiseSimilarity(task=args.task)
        set_default_seed()
        t0 = dt.datetime.now()
        if args.task == constants.CL:
            set_default_seed()
            compute_sim(data=all_train_samples, args=args, device=DEVICE,
                        outd=SIMS)
            set_default_seed()
        elif args.task == constants.SEG:
            # it has to be done differently. the similarity is measured
            # only between samples within the same class.

            for k in args.name_classes.keys():
                samples_in_same_class = [
                    sx for sx in all_train_samples if sx[3] == k]
                print("Computing similarities for class {}:".format(k))
                set_default_seed()
                compute_sim(data=samples_in_same_class, args=args,
                            device=DEVICE, outd=SIMS, label=k)
                set_default_seed()

        msg = "Time to compute sims {}: {}".format(
            tag_sims, dt.datetime.now() - t0
        )
        print(msg)
        log(training_log, msg)

        # compress, move files.

        if "CC_CLUSTER" in os.environ.keys():  # if CC
            cmdx = "cd {} && " \
                   "cd .. && " \
                   "tar -cf {}.tar.gz {} && " \
                   "cp {}.tar.gz {} && " \
                   "cd {} ".format(
                    SIMS,
                    tag_sims,
                    tag_sims,
                    tag_sims,
                    join(current_dir, "pairwise_sims"),
                    current_dir
                    )
        else:
            cmdx = "cd {} && " \
                   "tar -cf {}.tar.gz {} && " \
                   "cd {} ".format(
                    "./pairwise_sims",
                    tag_sims,
                    tag_sims,
                    current_dir
                    )

        tt = dt.datetime.now()
        print("Running bash-cmds: \n{}".format(cmdx.replace("&& ", "\n")))
        subprocess.run(cmdx, shell=True, check=True)
        msg += "\n time to run the command {}: {}".format(
            cmdx, dt.datetime.now() - tt)
        print(msg)
        log(training_log, msg)

    else:  # unzip if necessary.
        cmdx = None
        if "CC_CLUSTER" in os.environ.keys():  # if CC, copy to node.
            pr = join(placement_node, parent, "pairwise_sims")
            folder = join(pr, tag_sims)
            uncomp = False
            if not os.path.exists(folder):
                uncomp = True
            else:
                if len(os.listdir(folder)) == 0:
                    uncomp = True
            if uncomp:
                cmdx = "cp {}/{}.tar.gz {} && " \
                       "cd {} && " \
                       "tar -xf {}.tar.gz && " \
                       "cd {} ".format(
                            "./pairwise_sims",
                            tag_sims,
                            pr,
                            pr,
                            tag_sims,
                            current_dir
                            )

        else:
            folder = join('./pairwise_sims', tag_sims)
            uncomp = False
            if not os.path.exists(folder):
                uncomp = True
            else:
                if len(os.listdir(folder)) == 0:
                    uncomp = True

            if uncomp:
                cmdx = "cd {} && " \
                       "tar -xf {}.tar.gz && " \
                       "cd {} ".format(
                            "./pairwise_sims",
                            tag_sims,
                            current_dir
                            )

        if cmdx is not None:
            tt = dt.datetime.now()
            print("Running bash-cmds: \n{}".format(cmdx.replace("&& ", "\n")))
            subprocess.run(cmdx, shell=True, check=True)
            msg = "runtime of ALL the bash-cmds: {}".format(
                dt.datetime.now() - tt)
            print(msg)
            log(training_log, msg)

    return 0


def clear_rootpath(samples, args):
    """
    Remove the rootpath from the samples (img, mask) to be host independent.
    RETURNS A COPY OF THE SAMPLES UPDATED.

    :param samples: list of samples where each sample is a list. See format
    of samples for datasets.
        # 0. id: float, a unique id of the sample within the entire dataset.
        # 1. path_img: str, path to the image.
        # 2. path_mask: str or None, path to the mask if there is any.
        # Otherwise, None.
        # 3. label: int, the class label of the sample.
        # 4. tag: int in {0, 1, 2} where: 0: the samples belongs to the
        # supervised set (L). 1: The  sample belongs to the unsupervised set
        # (U). 2: The sample belongs to the set of newly labeled samples (
        # L'). This sample came from U and was labeling following a specific
        # mechanism.
    :param args: object. args of main.py.
    :return: a COPY list of samples (not inplace modification)
    """
    spls = deepcopy(samples)
    rootpath = get_rootpath_2_dataset(args)

    for i, sm in enumerate(spls):
        img_pth = sm[1]
        mask_pth = sm[2]
        l_pths = [img_pth, mask_pth]
        for j, pth in enumerate(l_pths):
            if pth:  # not None.
                pth = pth.replace(rootpath, '')
                if pth.startswith(os.path.sep):  # remove /
                    pth = pth[1:]

                l_pths[j] = pth

        img_pth, mask_pth = l_pths
        spls[i][1] = img_pth
        spls[i][2] = mask_pth

    return spls


def get_init_sup_samples(args,
                         sampler,
                         COMMON,
                         train_samples,
                         OUTD
                         ):
    """
    Get the initial full supervised data.
    :return:
    """
    previous_pairs = dict()
    previous_errors = False

    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.
    cnd_drop_n = (args.dataset == constants.CAM16)
    cnd_drop_n &= (args.al_type != constants.AL_WSL)

    # round 0
    cnd = (args.al_type not in [constants.AL_FULL_SUP, constants.AL_WSL])
    cnd &= (args.al_it == 0)

    if  cnd:
        # deterministic function with respect to the original seed.
        set_default_seed()
        train_samples = sampler.sample_init_random_samples(train_samples)
        set_default_seed()
        # store on disc: remove the rootpath from files to be host-independent.
        # store relative paths not absolute.
        base_f = 'train_{}.csv'.format(args.al_it)
        al_outf = join(COMMON, base_f)
        csv_writer(clear_rootpath(train_samples, args),
                   al_outf
                   )
        shutil.copyfile(al_outf, join(OUTD, base_f))

    # round > 0: combine all the samples of the previous al rounds
    # and the selected samples for this round.
    cnd = (args.al_type not in [constants.AL_FULL_SUP, constants.AL_WSL])
    cnd &= (args.al_it > 0)
    if cnd:
        # 'train_{i}.csv' contains the selected samples at round i.
        lfiles = [join(
            COMMON, 'train_{}.csv'.format(t)) for t in range(args.al_it + 1)]

        if (args.al_type == constants.AL_LP) and (args.task == constants.SEG):
            # load previous pairs:
            # previous pairs are pairs that have been pseudo-labeled in the
            # previous al round. they are ready to be used as
            # pseudo-segmented samples. no statistical constraints will be
            # applied on them.
            fz = join(COMMON, 'train_pairs_{}.pkl'.format(args.al_it - 1))
            with open(fz, 'rb') as fp:
                previous_pairs = pkl.load(fp)

        train_samples = []
        rootpath = get_rootpath_2_dataset(args)
        for fx in lfiles:
            # load using the current host-root-path.
            train_samples.extend(csv_loader(fx,
                                            rootpath,
                                            drop_normal=cnd_drop_n
                                            )
                                 )

        # Force: set all the samples in train_samples to L.
        for tt in range(len(train_samples)):
            train_samples[tt][4] = constants.L

        # ============== block to delete =======================================
        # in the case we skipped previous rounds because we restart the
        # code, if we are in cc and use node, the paths will not match
        # since they are built upon the job id. so, we need to change it.
        if "CC_CLUSTER" in os.environ.keys():
            for i in range(len(train_samples)):
                front = os.sep.join(train_samples[i][1].split(os.sep)[:3])
                cnd = (front != os.environ["SLURM_TMPDIR"])
                if cnd:
                    # update the image input path
                    train_samples[i][1] = train_samples[i][1].replace(
                        front, os.environ["SLURM_TMPDIR"]
                    )

                    if args.task == constants.SEG:
                        # update the mask path
                        train_samples[i][2] = train_samples[i][2].replace(
                            front, os.environ["SLURM_TMPDIR"]
                        )

                    previous_errors = True

            # TODO: remove the above block. no longer necessary.
            # since we use relative paths in the node, we shouldn't have
            # mismatching paths when restarting the code.
            assert not previous_errors, "ERROR."
        # ======================================================================

        set_default_seed()
        for i in range(100):
            random.shuffle(train_samples)
        set_default_seed()

    return train_samples, previous_pairs, previous_errors


def get_leftover(args,
                 train_csv,
                 rootpath,
                 train_samples
                 ):
    """
    Get the leftover samples.
    :return:
    """
    tr_leftovers = []  # the leftovers...
    ids_org = []  # ids of the entire trainset.
    ids_curt = []  # ids of the current trainset set (full sup only)
    tr_original = []  # samples of entire trainset

    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.
    cnd_drop_n = (args.dataset == constants.CAM16)
    cnd_drop_n &= (args.al_type != constants.AL_WSL)

    if args.al_type not in [constants.AL_FULL_SUP, constants.AL_WSL]:
        tr_original = csv_loader(train_csv, rootpath, drop_normal=cnd_drop_n)

        ids_org = [z[0] for z in tr_original]
        ids_curt = [z[0] for z in train_samples]
        tr_leftovers = []  # the leftovers...
        t0 = dt.datetime.now()
        for i, z in enumerate(ids_org):
            if z not in ids_curt:
                tr_leftovers.append(deepcopy(tr_original[i]))
        print("Searching took {}".format(dt.datetime.now() - t0))
        ids_leftovers = [z[0] for z in tr_leftovers]
        # Searching took 0:01:18.894629 for mnist 0 round. this can be done by
        # diff sets as well. but, we are not sure what set does to the order nor
        # the randomness. this is loop is safe.
        # tr_leftovers = [z for z in tr_original if z not in train_samples]

    return tr_leftovers, ids_org, ids_curt, tr_original


def prologue_opt(args,
                 OUTD_OPTMASKS,
                 SHARED_OPT_MASKS
                 ):
    """
    Create folders for optomization.
    :param args:
    :return:
    """
    # create sub-folders:
    subs = [
        "learning", "gifs", "tmp", "bin_masks", "continuous_masks",
        "final_masks"]
    for fd in subs:
        if not os.path.exists(join(OUTD_OPTMASKS, fd)):
            os.makedirs(join(OUTD_OPTMASKS, fd))

    # where to store the optimized binary masks as images for future use by
    # dataset.
    if args.share_masks:
        msg = "{} is not a valid directory.".format(SHARED_OPT_MASKS)
        assert os.path.isdir(SHARED_OPT_MASKS), msg
        fd_masks = SHARED_OPT_MASKS
    else:
        msg = "{} is not a valid directory.".format(OUTD_OPTMASKS)
        fd_masks = join(OUTD_OPTMASKS, "bin_masks")
        assert os.path.isdir(fd_masks), msg

    metrics_fd = join(fd_masks, "metrics")  # inside the fd of the masks.
    if not os.path.isdir(metrics_fd):
        os.makedirs(metrics_fd)


    return metrics_fd


def pair_samples(args,
                 train_samples,
                 tr_leftovers,
                 SIMS,
                 previous_pairs,
                 fd_p_msks
                 ):
    """
    Pair samples.
    :return:
    """
    VERBOSE = False

    acc_new_samples = 0.
    nbrx = 0
    train_samples_before_merging = deepcopy(train_samples)
    pairs = dict()
    metrics_fd = join(fd_p_msks, "metrics")  # inside the fd of the masks.

    if args.al_type == constants.AL_LP:
        set_default_seed()
        pairmaker = PairSamples(task=args.task,
                                knn=args.knn
                                )
        set_default_seed()
        announce_msg("starts pairing...")
        pairs = pairmaker(train_samples, tr_leftovers, SIMS)
        announce_msg("finishes pairing...")
        set_default_seed()

        # if pair exists in the previous round, delete it.
        # pairs must contain only newly paired samples or samples that have
        # been paired but changed their source.

        for k in pairs.keys():
            if (k, pairs[k]) in previous_pairs:  # same pair exist.
                pairs.pop(k)

    return pairs, acc_new_samples, nbrx,train_samples_before_merging


def merge_pairs_tr_samples(args,
                           some_pairs,
                           tr_original,
                           ids_org,
                           train_samples
                           ):
    """
    merge some pairs intro the trainset.
    :return:
    """
    set_default_seed()

    acc_new_samples = 0.
    train_samples = deepcopy(train_samples)

    if args.al_type != constants.AL_LP:
        return train_samples, acc_new_samples


    # add the paired samples to the trainset: previous only.
    new_samples = []

    for k in list(some_pairs.keys()):
        # pairs: dict of: k (key: id of unlabeled sample): val (value: id
        # of labeled sample)
        idtoadd = k  # pairs[k]
        stoadd = deepcopy(tr_original[ids_org.index(idtoadd)])

        # previous pairs.
        acc_new_samples += (tr_original[ids_org.index(
            some_pairs[k])][3] == stoadd[3]) * 1.
        stoadd[4] = constants.PL  # to filter samples in the loss.
        # set the previously paired to pseudo-labeled.
        stoadd[3] = tr_original[ids_org.index(some_pairs[k])][3]
        # image-level label propagation. propagated labels are not
        # perfect.

        if args.task == constants.SEG:
            msg = "in weak.sup. setup, paired samples must have same " \
                  "image level label."
            cndx = tr_original[ids_org.index(some_pairs[k])][3] == stoadd[3]
            assert cndx, msg


        new_samples.append(stoadd)

    train_samples.extend(new_samples)  # add samples.
    # shuffle very well to mix pairs (new, old) with full sup.
    set_default_seed()
    for i in range(1000):
        random.shuffle(train_samples)
    set_default_seed()


    return train_samples, acc_new_samples


def get_trainset(args,
                 train_samples,
                 transform_tensor,
                 train_transform_img,
                 check_ps_msk_path,
                 previous_pairs,
                 fd_p_msks
                 ):
    """
    Get the trainset.
    :return:
    """
    set_default_seed()

    trainset = PhotoDataset(
        train_samples,
        args.dataset,
        args.name_classes,
        transform_tensor,
        set_for_eval=False,
        transform_img=train_transform_img,
        resize=None,
        resize_h_to=None,
        resize_mask=False,
        crop_size=args.crop_size,
        padding_size=(args.padding_ratio, args.padding_ratio),
        padding_mode=args.padding_mode,
        up_scale_small_dim_to=args.up_scale_small_dim_to,
        do_not_save_samples=True,
        ratio_scale_patch=args.ratio_scale_patch,
        for_eval_flag=False,
        scale_algo=args.scale_algo,
        enhance_color=False,
        enhance_color_fact=1.,
        check_ps_msk_path=check_ps_msk_path,
        previous_pairs=previous_pairs,
        fd_p_msks=fd_p_msks
    )

    set_default_seed()

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=_init_fn,
        collate_fn=default_collate
    )

    set_default_seed()

    out = {
        'trainset': trainset,
        'train_loader': train_loader
    }

    return trainset, train_loader


def get_validationset(args,
                      valid_samples,
                      transform_tensor,
                      padding_size_eval,
                      batch_size=None
                      ):
    """
    Get the validation set
    :param batch_size: int or None. batch size. if None, the value defined in
    `args.valid_batch_size` will be used.
    :return:
    """
    set_default_seed()
    validset = PhotoDataset(
        valid_samples,
        args.dataset,
        args.name_classes,
        transform_tensor,
        set_for_eval=False,
        transform_img=None,
        resize=None,
        resize_h_to=None,
        resize_mask=False,
        crop_size=None,
        padding_size=padding_size_eval,
        padding_mode=None if (padding_size_eval == (None, None)) else
        args.padding_mode,
        up_scale_small_dim_to=args.up_scale_small_dim_to,
        do_not_save_samples=True,
        ratio_scale_patch=args.ratio_scale_patch,
        for_eval_flag=True,
        scale_algo=args.scale_algo,
        enhance_color=False,
        enhance_color_fact=1.,
        check_ps_msk_path=False,
        fd_p_msks=None
    )
    set_default_seed()

    # we need more workers since the batch size is 1, and set_for_eval is
    # False (need more time to prepare a sample).
    valid_loader = DataLoader(
        validset,
        batch_size=args.valid_batch_size if batch_size is None else batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        worker_init_fn=_init_fn
    )
    set_default_seed()

    out = {
        'validset': validset,
        'valid_loader': valid_loader
    }

    return validset, valid_loader


def get_dataset_for_pseudo_anno(args,
                                new_samples,
                                transform_tensor,
                                padding_size_eval
                                ):
    """
    Get dataset for pseudo-annotation.
    :return:
    """
    set_default_seed()
    trainset_eval = PhotoDataset(
        new_samples,
        args.dataset,
        args.name_classes,
        transform_tensor,
        set_for_eval=False,
        transform_img=None,
        resize=None,
        resize_h_to=None,
        resize_mask=False,
        crop_size=None,
        padding_size=padding_size_eval,
        padding_mode=None if (padding_size_eval == (None, None)) else
        args.padding_mode,
        up_scale_small_dim_to=args.up_scale_small_dim_to,
        do_not_save_samples=True,
        ratio_scale_patch=args.ratio_scale_patch,
        for_eval_flag=True,
        scale_algo=args.scale_algo,
        enhance_color=False,
        enhance_color_fact=1.,
        check_ps_msk_path=False,
        fd_p_msks=None
    )

    set_default_seed()
    train_eval_loader = DataLoader(
        trainset_eval,
        batch_size=args.valid_batch_size if args.task == constants.CL else 1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=default_collate,
        worker_init_fn=_init_fn
    )

    set_default_seed()

    return trainset_eval, train_eval_loader


def recurrent_plotting(vl_stats,
                       OUTD_VL,
                       tr_stats,
                       OUTD_TR,
                       CRITERION,
                       OUTD_TLB,
                       args,
                       PLOT_STATS,
                       epoch,
                       plot_freq,
                       force
                       ):
    """
    Do some recurrent plotting.
    :return:
    """

    cnd = PLOT_STATS and (epoch % plot_freq == 0)
    if cnd or force:
        plot_curves_from_dict(vl_stats,
                              join(OUTD_VL.folder, "validset-stats.png"),
                              title="Validset stats. {}".format(args.loss),
                              dpi=100, plot_avg=True,
                              avg_perd=10
                              )

        plot_curves_from_dict(tr_stats,
                              join(OUTD_TR.folder, "trainset-stats.png"),
                              title="Trainset stats. {}".format(args.loss),
                              dpi=100,
                              plot_avg=True,
                              avg_perd=10
                              )


    cnd &= (CRITERION.t_tracker != [])
    cnd &= force
    if cnd:
        title = "t evolution. min: {}. max: {}.".format(
            min(CRITERION.t_tracker), max(CRITERION.t_tracker))

        plot_curve(CRITERION.t_tracker,
                   join(OUTD_TLB.folder, "tlb-evolution.png"),
                   title,
                   "epochs",
                   "t",
                   dpi=100
                   )


def clean_shared_masks(args,
                       SHARED_OPT_MASKS,
                       metrics_fd,
                       all_pairs
                       ):
    """
    Clean the folder shared_masks from duplicates.

    :param args:
    :param SHARED_OPT_MASKS:
    :param metrics_fd:
    :param all_pairs:
    :return:
    """
    if not args.share_masks:
        return 0

    # do some cleaning of the shared folder of masks.
    # some unlabeled samples are labeled. so, the files are no
    # longer useful. + their metrics
    l_pm_needed = [
        join(SHARED_OPT_MASKS, "{}-{}.bmp".format(
            id_u, all_pairs[id_u])) for id_u in all_pairs.keys()
    ]
    l_pm_exist = find_files_pattern(fd_in_=SHARED_OPT_MASKS, pattern_="*.bmp")
    set_default_seed()
    l_pm_del = list(set(l_pm_exist) - set(l_pm_needed))
    set_default_seed()

    l_mtr_needed = [
        join(metrics_fd, "{}-{}.pkl".format(
            id_u, all_pairs[id_u])) for id_u in all_pairs.keys()
    ]
    l_mtr_exist = find_files_pattern(fd_in_=metrics_fd, pattern_="*.pkl")
    set_default_seed()
    l_mtr_del = list(set(l_mtr_exist) - set(l_mtr_needed))
    set_default_seed()

    [os.remove(path) for path in l_pm_del + l_mtr_del]

    return 0


def seedj(epoch, j, cycle, conts):
    """
    Return a seed.
    :param epoch: int. epoch.
    :param j: int.
    :param cycle: int. cycle.
    :param conts: int. constant.
    :return:
    """
    return int(os.environ["MYSEED"]) + (epoch + j) * conts + (cycle + 1) * 10


def new_fresh_model(args, DEVICE):
    """
    Reload a new-fresh model.
    useful to restart training with certainty that the model is at the same
    exact state at t=0.
    :param args: object.
    :param DEVICE: torch device.
    :return:
    """
    ALLOW_MULTIGPUS = check_if_allow_multgpu_mode()
    NBRGPUS = torch.cuda.device_count()
    # ========================= Instantiate models =============================
    model = instantiate_models(args)

    # Check if we are using a user specific pre-trained model other than our
    # pre-defined pre-trained models. This can be used to to EVALUATE a
    # trained model. You need to set args.max_epochs to -1 so no training is
    # performed. This is a hack to avoid creating other function to deal with
    # LATER-evaluation after this code is done. This script is intended for
    # training. We evaluate at the end. However, if you missed something
    # during the training/evaluation (for example plot something over the
    # predicted images), you do not need to re-train the model. You can 1.
    # specify the path to the pre-trained model. 2. Set max_epochs to -1. 3.
    # Set strict to True. By doing this, we load the pre-trained model, and,
    # we skip the training loop, fast-forward to the evaluation.

    if args.model['path_pre_trained'] not in [None, 'None']:
        warnings.warn("You have asked to load a specific pre-trained "
                      "model from {} .... [OK]".format(
            args.model['path_pre_trained']))

        model = load_pre_pretrained_model(
            model=model,
            path_file=args.model['path_pre_trained'],
            strict=args.model['strict'],
            freeze_classifier=args.freeze_classifier
        )

    # Check if there are multiple GPUS.
    if ALLOW_MULTIGPUS:
        model = MyDataParallel(model)
        if args.batch_size < NBRGPUS:
            warnings.warn("You asked for MULTIGPU mode. However, "
                          "your batch size {} is smaller than the number of "
                          "GPUs available {}. This is fine in practice. "
                          "However, some GPUs will be idol. "
                          "This is just a warning .... "
                          "[OK]".format(args.batch_size, NBRGPUS))
    model.to(DEVICE)
    # freeze the classifier if needed
    if args.freeze_classifier:
        warnings.warn("You asked to freeze the classifier."
                      "We are going to do it right now.")
        model.freeze_cl()
        assert model.assert_cl_is_frozen(), "Something is wrong"
    # ==========================================================================

    return model