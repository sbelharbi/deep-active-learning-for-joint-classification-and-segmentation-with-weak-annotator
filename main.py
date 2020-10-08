import argparse
import os
from os.path import join
from copy import deepcopy
import datetime as dt
import warnings
import sys
import subprocess
import shutil
import pickle as pkl


import torch


from deeplearning.train import train_one_epoch
from deeplearning.train import validate
from deeplearning.train import final_validate
from deeplearning.train import pseudo_annotate_some_pairs
from deeplearning.train import estimate_best_seg_thres


from tools import log
from tools import get_device
from tools import get_rootpath_2_dataset
from tools import get_cpu_device
from tools import get_transforms_tensor
from tools import get_train_transforms_img
from tools import copy_model_state_dict_from_gpu_to_cpu


from shared import check_if_allow_multgpu_mode
from shared import announce_msg
from shared import csv_writer
from shared import csv_loader
from shared import CONST1

import constants


from prologues import prologue_init
from prologues import check_if_round_already_done
from prologues import prologue_fds_0
from prologues import prologue_fds_1
from prologues import get_csv_files
from prologues import compute_similarities
from prologues import get_init_sup_samples
from prologues import get_leftover
from prologues import pair_samples
from prologues import prologue_opt
from prologues import merge_pairs_tr_samples
from prologues import get_trainset
from prologues import get_validationset
from prologues import recurrent_plotting
from prologues import clean_shared_masks
from prologues import seedj
from prologues import new_fresh_model
from prologues import clear_rootpath


from instantiators import instantiate_optimizer
from instantiators import instantiate_loss
from instantiators import instantiate_sampler


from vision import PlotActiveLearningRounds

from parseit import parse_input


from reproducibility import reset_seed
from reproducibility import set_default_seed


# args.num_workers * this_factor. Useful when setting set_for_eval to False,
# batch size =1.
FACTOR_MUL_WORKERS = 2
# and we are in an evaluation mode (to go faster and coop with the lag
# between the CPU and GPU).

# Can be activated only for "Caltech-UCSD-Birds-200-2011" or
# "Oxford-flowers-102"
DEBUG_MODE = False
# dataset to go fast. If True, we select only few samples for training
# , validation, and test.

FRQ = 2.  # number of times to plot the train figures.

# =============================================
# Parse the inputs and deal with the yaml file.
# init. reproducibility.
# =============================================
args, args_dict, input_args = parse_input()

NBRGPUS = torch.cuda.device_count()

ALLOW_MULTIGPUS = check_if_allow_multgpu_mode()


if __name__ == "__main__":
    init_time = dt.datetime.now()

    # ==================================================
    # Device, criteria, folders, output logs, callbacks.
    # ==================================================

    DEVICE = get_device(args)
    CPUDEVICE = get_cpu_device()

    CRITERION = instantiate_loss(args).to(DEVICE)

    # plot freq.
    # plot only if we are outside cc.
    PLOT_STATS = not ('CC_CLUSTER' in os.environ.keys())
    plot_freq = max(args.max_epochs  - 1, int(args.max_epochs / FRQ))

    output_fds = prologue_init(args)
    COMMON = output_fds['COMMON']
    OUTD = output_fds['OUTD']
    VISION = output_fds['VISION']
    SHARED_OPT_MASKS = output_fds['SHARED_OPT_MASKS']
    ROUNDS = output_fds['ROUNDS']
    OUTD = output_fds['OUTD']
    tag_sims = output_fds['tag_sims']
    SIMS = output_fds['SIMS']
    tag = output_fds['tag']
    placement_scr = output_fds['placement_scr']
    parent = output_fds['parent']
    exp_name = output_fds['exp_name']
    placement_node = output_fds['placement_node']


    out_0 = prologue_fds_0(args,
                           OUTD,
                           COMMON,
                           SIMS,
                           VISION,
                           ROUNDS,
                           SHARED_OPT_MASKS
                           )

    OUTD_TR = out_0['OUTD_TR']
    OUTD_VL = out_0['OUTD_VL']
    OUTD_TS = out_0['OUTD_TS']
    OUTD_TLB = out_0['OUTD_TLB']
    OUTD_LO = out_0['OUTD_LO']
    OUTD_OPTMASKS = out_0['OUTD_OPTMASKS']
    fd_p_msks = out_0['fd_p_msks']

    # ==========================================================================
    #                      CHECK IF WE HAVE TO STOP AL
    #             BECAUSE THERE WERE NO SAMPLES LEFT TO SELECT FROM
    #                        IN THE PREVIOUS ROUND.
    # ==========================================================================
    if os.path.isfile(join(COMMON, 'stop_active_learning.txt')):
        warnings.warn("Exiting because there were no samples to select "
                      "from in the previous round.")
        sys.exit(0)
    # ==========================================================================

    # ==========================================================================
    #                 CHECK IF THIS ROUND HAS ALREADY BEEN PROCESSED.
    #           THIS CAN BE HELPFUL WHEN RESTARTING ALL ROUNDS DUE TO SOME
    #                 ERROR AT SOME SPECIFIC ROUND OR BECAUSE
    #                  THIS ROUND HAS ALREADY BEN PROCESSED.
    # ==========================================================================

    check_if_round_already_done(placement_scr,
                                parent,
                                tag,
                                exp_name,
                                OUTD,
                                VISION,
                                COMMON,
                                SHARED_OPT_MASKS
                                )

    training_log, results_log = prologue_fds_1(args,
                                               OUTD,
                                               input_args,
                                               args_dict
                                               )

    log(training_log, "\n\n ########### Training #########\n\n")
    log(results_log, "\n\n ########### Results #########\n\n")

    # ==========================================================
    # Data transformations: on PIL.Image.Image and torch.tensor.
    # ==========================================================

    train_transform_img = get_train_transforms_img(args)
    transform_tensor = get_transforms_tensor(args)

    # ==========================================================================
    # Datasets: load csv, datasets: train, valid, test.
    # ==========================================================================

    announce_msg("SPLIT: {} \t FOLD: {}".format(args.split, args.fold))

    train_csv, valid_csv, test_csv = get_csv_files(args)

    rootpath = get_rootpath_2_dataset(args)

    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.
    cnd_drop_n = (args.dataset == constants.CAM16)
    cnd_drop_n &= (args.al_type != constants.AL_WSL)

    train_samples = csv_loader(train_csv, rootpath, drop_normal=cnd_drop_n)
    valid_samples = csv_loader(valid_csv, rootpath, drop_normal=cnd_drop_n)
    test_samples = csv_loader(test_csv, rootpath, drop_normal=cnd_drop_n)

    # remove normal from name classes.
    if cnd_drop_n:
        args.name_classes.pop("normal")


    announce_msg("len original trainset: {}".format(len(train_samples)))


    # ==========================================================================
    #       START: PREPARE DATA FOR THE CURRENT ACTIVE LEARNING ROUND
    #                 BASED ON THE PREVIOUS ROUNDS.
    # ==========================================================================
    # adjust p_samples to a fixed int value across all the AL rounds.
    p_samples_backup = args.p_samples

    if args.dataset == constants.CAM16:
        vx = args.p_samples * len(train_samples) / 100.
        args.p_samples = round(vx) if vx < 1 else int(vx)
    else:
        args.p_samples = int(args.p_samples * len(train_samples) / 100.)

    set_default_seed()
    sampler = None
    cnd = (args.al_type not in [constants.AL_FULL_SUP, constants.AL_WSL])
    if cnd:
        sampler = instantiate_sampler(args=args, device=DEVICE)
    set_default_seed()
    previous_errors = False  # if False, it means that there was no error in
    # computing previous rounds. if True, it means that we are restarting the
    # code. previous rounds are ok, but at some round, there was an error.
    # instead of recomputing every round, we skip the fine ones, and start
    # the one where there was an error. We can detect if there was an error
    # in these rounds if the id of the job has changed. if every round was
    # processed fine, all of the samples should have the same job id in their
    # name as the current job id. if the id has changed, this means that the
    # job has been restarted.

    # ==========================================================================
    #         GET THE FULL SUPERVISED DATA ONLY FROM THE PREVIOUS ROUNDS
    # ==========================================================================


    # ==========================================================================
    #                 START: DEAL WITH OUR METHOD'S COMPUTATIONS
    #                    1. COMPUTE SIMILARITIES
    #                    2. DO LABEL PROPAGATION (PAIRING)
    # ==========================================================================

    # 1. COMPUTE SIMILARITIES
    compute_similarities(args,
                         tag_sims,
                         train_csv,
                         rootpath,
                         DEVICE,
                         SIMS,
                         training_log,
                         placement_node,
                         parent
                         )

    # 2. get samples
    train_samples, previous_pairs, previous_errors = get_init_sup_samples(
        args,
        sampler,
        COMMON,
        train_samples,
        OUTD
        )

    tr_leftovers, ids_org, ids_curt, tr_original = get_leftover(args,
                                                                train_csv,
                                                                rootpath,
                                                                train_samples
                                                                )

    check_ps_msk_path = (args.task == constants.SEG)
    check_ps_msk_path &= (args.al_type == constants.AL_LP)

    # 3. Label propagation: pair samples.
    metrics_fd = prologue_opt(args,
                              OUTD_OPTMASKS,
                              SHARED_OPT_MASKS
                              )

    pairs, acc_new_samples, nbrx, train_samples_before_merging = pair_samples(
        args,
        train_samples,
        tr_leftovers,
        SIMS,
        previous_pairs,
        fd_p_msks
        )

    train_samples, acc_new_samples = merge_pairs_tr_samples(args,
                                                            previous_pairs,
                                                            tr_original,
                                                            ids_org,
                                                            train_samples
                                                            )

    # total number of pseudo-labeled samples.
    # keys in pairs may exist in previous_pairs due to proximity change.
    set_default_seed()
    ids_all_pairs = set(list(pairs.keys()) + list(previous_pairs.keys()))
    ids_all_pairs = list(ids_all_pairs)
    ids_all_pairs.sort()  # to make the code deterministic. set is not.
    all_pairs = dict()
    for k in ids_all_pairs:
        if k in pairs.keys():
            all_pairs[k] = pairs[k]  # take the new pair.
        else:
            all_pairs[k] = previous_pairs[k]

    nbrx = len(ids_all_pairs)
    set_default_seed()

    if nbrx != 0:
        acc_new_samples = 100. * acc_new_samples / float(nbrx)
    msg = "Accuracy of the set of samples using the " \
          "propagated label: {}%. " \
          "Number of propagated samples: {}/{}.".format(
        acc_new_samples, nbrx, len(tr_leftovers))

    print(msg)
    log(results_log, msg)

    propagation = {'acc': acc_new_samples,
                  'nbr_samples_prop': nbrx,
                  'nbr_fully_sup': len(train_samples_before_merging),
                  'leftovers': tr_leftovers,
                  'args': deepcopy(vars(args)),  # args may change later.
                  'p_samples': p_samples_backup
                  }


    # ==========================================================================
    #                 END: DEAL WITH OUR METHOD'S COMPUTATIONS
    # ==========================================================================

    # store all the training samples.
    base_f = 'all_train_{}.csv'.format(args.al_it)
    al_outf = join(COMMON, base_f)
    csv_writer(clear_rootpath(train_samples, args),
               al_outf
               )
    shutil.copyfile(al_outf, join(OUTD, base_f))
    # ==========================================================================
    #                 END: PREPARE DATA FOR THE CURRENT ACTIVE LEARNING ROUND
    #                 BASED ON THE PREVIOUS ROUNDS.
    # ==========================================================================

    if args.al_type == constants.AL_FULL_SUP:
        train_samples = csv_loader(train_csv, rootpath, drop_normal=cnd_drop_n)
        for i, _ in enumerate(train_samples):
            train_samples[i][4] = constants.L  # set to supervised.

    if args.al_type == constants.AL_WSL:
        train_samples = csv_loader(train_csv, rootpath, drop_normal=cnd_drop_n)
        for i, _ in enumerate(train_samples):
            train_samples[i][4] = constants.U  # set to unlabeled.

    msg = "ActiveL: {}, round: {}, TR-Samples: {}. FULL-SUP-Samples: " \
          "{}. Propagated-Samples: {}. Dataset: {}. ".format(
        args.al_type,
        args.al_it,
        len(train_samples),
        len(train_samples_before_merging),
        nbrx,
        args.dataset
    )
    announce_msg(msg)
    log(results_log, msg)
    log(training_log, msg)

    if args.weight_pseudo_loss and (args.al_type == constants.AL_LP) and (
            nbrx > 0):
        CRITERION.set_weight_pl(val=(1./float(nbrx)))

    announce_msg("creating datasets and dataloaders")

    trainset, train_loader = get_trainset(args,
                                          train_samples,
                                          transform_tensor,
                                          train_transform_img,
                                          check_ps_msk_path,
                                          previous_pairs,
                                          fd_p_msks
                                          )

    padding_size_eval = (args.padding_ratio,
                         args.padding_ratio) if args.pad_eval else (None, None)

    validset, valid_loader = get_validationset(args,
                                               valid_samples,
                                               transform_tensor,
                                               padding_size_eval
                                               )

    model = new_fresh_model(args, DEVICE)

    # Copy the model's params.
    best_state_dict = deepcopy(model.state_dict())  # it has to be deepcopy.

    with open(join(OUTD, 'nbr_params.txt'), 'w') as fend:
        fend.write("Model: {}. \n NBR-params: {}.".format(
            model, model.get_nbr_params()
        ))

    # ========================== INSTANTIATE OPTIMIZER =========================
    set_default_seed()

    optimizer, lr_scheduler = instantiate_optimizer(args, model)
    if model.freeze_classifier:
        assert model.assert_cl_is_frozen(), "Something is wrong"

    # ============================ TRAINING ==============================
    set_default_seed()
    tr_stats, vl_stats = None, None

    best_val_metric = None
    best_epoch = 0
    best_cycle = 0

    # validate before start training so curves start from the same point.
    # this value will not be counted as a point where to pick a best model
    set_default_seed()
    vl_stats = validate(model,
                        validset,
                        valid_loader,
                        CRITERION,
                        DEVICE,
                        vl_stats,
                        args,
                        epoch=-1,
                        log_file=training_log
                        )
    set_default_seed()

    announce_msg("start training")
    tx0 = dt.datetime.now()

    set_default_seed()

    epoch = 0
    cycle = 0
    while epoch < args.max_epochs:
        # debug.

        # if (epoch == 0) and (cycle == 0):
        #     epoch = args.max_epochs - 1

        # reseeding tr/vl samples.
        reset_seed(seedj(epoch, 1, cycle, CONST1))
        trainset.set_up_new_seeds()
        reset_seed(seedj(epoch, 2, cycle, CONST1))

        tr_stats= train_one_epoch(model,
                                  optimizer,
                                  train_loader,
                                  CRITERION,
                                  DEVICE,
                                  tr_stats,
                                  args,
                                  trainset,
                                  epoch,
                                  cycle,
                                  training_log,
                                  ALLOW_MULTIGPUS=ALLOW_MULTIGPUS,
                                  NBRGPUS=NBRGPUS
                                  )

        # add learning rate to the plots.
        if lr_scheduler:
            curt_lr = lr_scheduler.get_last_lr()[-1]  # assume only 1 group.
        else:
            curt_lr = args.optimizer['optn0__lr']

        if 'lr' in tr_stats.keys():
            tr_stats['lr'].append(curt_lr)
        else:
            tr_stats['lr'] = [curt_lr]  # assume only 1 group.

        if model.freeze_classifier:
            assert model.assert_cl_is_frozen(), "Something is wrong"

        if lr_scheduler:  # for Pytorch > 1.1 : opt.step() then l_r_s.step().
            lr_scheduler.step(epoch)

        # Eval validation set.
        vl_stats = validate(model,
                            validset,
                            valid_loader,
                            CRITERION,
                            DEVICE,
                            vl_stats,
                            args,
                            epoch=epoch,
                            cycle=cycle,
                            log_file=training_log
                            )

        # validation metrics: selection criterion is based on classification
        # accuracy and Dice index over the validation set.
        if args.subtask == constants.SUBCLSEG:
            metric_val = vl_stats["acc"][-1] + vl_stats["dice_idx"][-1]
        elif args.subtask == constants.SUBCL:
            metric_val = vl_stats["acc"][-1]
        elif args.subtask == constants.SUBSEG:
            metric_val = vl_stats["dice_idx"][-1]

        cndvl = (args.task == constants.SEG)
        cndvl &= (args.subtask in [constants.SUBCLSEG, constants.SUBSEG])
        cndvl &= args.freeze_classifier
        if cndvl:
            metric_val = vl_stats["dice_idx"][-1]

        if (best_val_metric is None) or (metric_val >= best_val_metric):
            # best_val_loss:
            best_val_metric = metric_val
            # it has to be deepcopy.
            best_state_dict = deepcopy(model.state_dict())
            # Expensive operation: disc I/O.
            # torch.save(best_model.state_dict(), join(OUTD, "best_model.pt"))
            best_epoch = epoch
            best_cycle = cycle
            print("BEST: vl acc: {:.2f}%. "
                  "vl dice {:.2f}%. "
                  "CYCLE: {}.".format(vl_stats["acc"][-1] * 100.,
                                      vl_stats["dice_idx"][-1] * 100.,
                                      best_cycle
                                      )
                  )

        # update t of elb at each epoch.
        CRITERION.update_t()
        CRITERION.update_scales()

        recurrent_plotting(vl_stats,
                           OUTD_VL,
                           tr_stats,
                           OUTD_TR,
                           CRITERION,
                           OUTD_TLB,
                           args,
                           PLOT_STATS,
                           epoch,
                           plot_freq,
                           force=False
                           )

        epoch += 1

        # end of current iteration =============================================

        # ======================================================================
        #                    RESTART THE LEARNING WHILE:
        #            - PSEUDO-LABEL ALL PAIRS (NEW AND OLD).
        #            - USING THE PSEUDO-MASKS OF THE NEWLY PAIRED SAMPLES.
        #                        --------------
        #  WHY USING PSEUDO-MASKS OF NEWLY-PAIRED SAMPLES DURING THIS CURRENT
        #  AL ROUND? EITHER WE WAIT UNTIL THE NEXT AL ROUND TO USE THEM,
        #  OR EXPLOIT THEM IN THIS CURRENT AL ROUND. WE WENT WITH THE SECOND
        #  OPTION SINCE IT ALLOWS TO BENEFIT FROM THEM RIGHT NOW INSTEAD OF
        #  WAITING FOR THE NEXT ROUND. THE COST? WE RUN TWICE THE LEARNING.
        #  IN REAL-APPLICATIONS, THIS IS THE BEST CHOICE.
        # ======================================================================
        # conditions to re-start the learning.
        cnd = (epoch == args.max_epochs)
        cnd &= (args.al_type == constants.AL_LP)
        cnd &= (args.task == constants.SEG)
        cnd &= (args.subtask in [constants.SUBSEG, constants.SUBCLSEG])
        cnd &= (len(all_pairs.keys()) > 0)

        if (epoch == args.max_epochs) and cnd:
            pass
        elif (epoch == args.max_epochs) and (not cnd):  # if conditions are not
            # met, leave.
            break
        elif epoch < args.max_epochs:
            continue  # do not run the below code.

        if cycle == 1:  # get out if we already did one cycle of the below code.
            break

        announce_msg("Going to restart the training....")

        # prepare to continue learning.
        set_default_seed()

        # 1. estimate the segmentation threshold if needed. only pseudo-labeled
        # samples use a an estimated threshold (if requested).

        threshold = args.seg_threshold

        # set the model to its best one (found using validation)
        model.load_state_dict(best_state_dict)

        if args.estimate_pseg_thres:
            zout = estimate_best_seg_thres(model,
                                           validset,
                                           valid_loader,
                                           CRITERION,
                                           DEVICE,
                                           args,
                                           epoch=epoch,
                                           cycle=cycle,
                                           log_file=training_log
                                           )
            for zk in zout.keys():
                propagation['vl_init_' + zk] = zout[zk]

            threshold = zout['best_threshold']


        propagation["pseudo_seg_thres_init"] = threshold
        log(results_log, "SEG-THRESHOLD-INIT: {} [EST-VL: {}]".format(
            threshold, args.estimate_pseg_thres))

        # 2. pseudo-annotate all the samples.
        new_paired_samples = [
            deepcopy(ssx) for ssx in tr_original if ssx[0] in list(
                all_pairs.keys())]
        # force them to be labeled
        for iz, el in enumerate(new_paired_samples):
            el[4] = constants.L  # in-place update.

        announce_msg("Pseudo-labeling all-pairs ({}). "
                     "cycle {}".format(len(list(pairs.keys())), cycle))
        outps = pseudo_annotate_some_pairs(model,
                                           all_pairs,
                                           new_paired_samples,
                                           args,
                                           CRITERION,
                                           transform_tensor,
                                           padding_size_eval,
                                           DEVICE,
                                           fd_p_msks,
                                           SHARED_OPT_MASKS,
                                           threshold,
                                           results_log,
                                           "SEG-PL-INIT"
                                           )

        propagation["sum_dice_new_p"] = outps['sum_dice']
        propagation["number_new_p"] = outps['n_samples']

        # ===== reset the training.

        # 2. use the pseudo-masks of all paired samples.

        train_samples , _ = merge_pairs_tr_samples(
            args=args,
            some_pairs=all_pairs,
            tr_original=tr_original,
            ids_org=ids_org,
            train_samples=train_samples_before_merging
            )

        # get a new trainset
        trainset, train_loader = get_trainset(args,
                                              train_samples,
                                              transform_tensor,
                                              train_transform_img,
                                              check_ps_msk_path,
                                              all_pairs,
                                              fd_p_msks
                                              )

        # reset the model to the initial parameters
        del model
        model = new_fresh_model(args, DEVICE)

        # create optimizer.
        optimizer, lr_scheduler = instantiate_optimizer(args, model)
        if model.freeze_classifier:
            assert model.assert_cl_is_frozen(), "Something is wrong"

        # instantiate training loss.
        CRITERION = instantiate_loss(args).to(DEVICE)
        if args.weight_pseudo_loss and (args.al_type == constants.AL_LP) and (
                nbrx > 0):
            CRITERION.set_weight_pl(val=(1. / float(nbrx)))

        # to get out of this loop the next time.
        cycle += 1

        # reset epoch
        epoch = 0
        # DO NOT RESET best_state_dict and best_val_metric. cycle=1 needs to
        # find a better model than cycle=0.

        # sys.exit()


    # ==========================================================================
    #                     DO CLOSING-STUFF BEFORE LEAVING
    # ==========================================================================
    # Measure performance using the best model over: train/valid/test sets.
    # Train set: needs to reload it with eval-transformations,
    # not train-transformations.

    # do the final plotting
    recurrent_plotting(vl_stats,
                       OUTD_VL,
                       tr_stats,
                       OUTD_TR,
                       CRITERION,
                       OUTD_TLB,
                       args,
                       PLOT_STATS,
                       epoch,
                       plot_freq,
                       force=True
                       )

    # Reset the models parameters to the best found ones. ++++++++++++++++++++++
    model.load_state_dict(best_state_dict)

    threshold = args.seg_threshold
    cnd = args.estimate_seg_thres
    cnd |= ((args.al_type == constants.AL_LP) and args.estimate_pseg_thres)
    if cnd:
        zout22 = estimate_best_seg_thres(model,
                                         validset,
                                         valid_loader,
                                         CRITERION,
                                         DEVICE,
                                         args,
                                         epoch=epoch,
                                         cycle=cycle,
                                         log_file=training_log
                                         )
        for zk in zout22.keys():
            propagation['vl_final_pred_' + zk] = zout22[zk]

        threshold = zout22['best_threshold']
        if args.estimate_seg_thres:
            args.seg_threshold = threshold

    propagation["pseudo_seg_thres_final_pred"] = threshold

    log(results_log, "Loss: {} \n"
                     "Best epoch: {} "
                     "Best cycle: {}".format(args.loss, best_epoch, best_cycle))

    # We need to do each set sequentially to free the memory.
    msg = "End training. Time: {}".format(dt.datetime.now() - tx0)

    announce_msg(msg)
    log(training_log, msg)

    # Save train statistics (train, valid)
    stats_to_dump = {
        "train": tr_stats,
        "valid": vl_stats
    }
    with open(join(OUTD, "train_stats.pkl"), "wb") as fout:
        pkl.dump(stats_to_dump, fout, protocol=pkl.HIGHEST_PROTOCOL)

    # store propagation info.
    propagation['best_epoch'] = best_epoch
    propagation['best_cycle'] = best_cycle
    with open(join(OUTD, 'propagation.pkl'), 'wb') as fpp:
        pkl.dump(propagation, fpp, protocol=pkl.HIGHEST_PROTOCOL)

    tx0 = dt.datetime.now()

    set_default_seed()
    msg = "start final processing stage \n" \
          "Best epoch: {} " \
          "Best cycle: {}".format(best_epoch, best_cycle)
    announce_msg(msg)
    log(training_log, msg)

    del trainset
    del train_loader

    # ==========================================================================
    #                               VALIDATION SET
    # ==========================================================================
    set_default_seed()
    final_validate(model,
                   valid_loader,
                   CRITERION,
                   DEVICE,
                   validset,
                   OUTD_VL.folder,
                   args,
                   log_file=results_log,
                   name_set="validset",
                   store_results=False,
                   apply_selection_tech=False
                   )


    # ==========================================================================
    #                               TEST SET
    # ==========================================================================
    testset, test_loader = get_validationset(args,
                                             test_samples,
                                             transform_tensor,
                                             padding_size_eval,
                                             batch_size=None
                                             )

    final_validate(model,
                   test_loader,
                   CRITERION,
                   DEVICE,
                   testset,
                   OUTD_TS.folder,
                   args,
                   log_file=results_log,
                   name_set="testset",
                   store_results=True,  # too heavy to store without benefit
                   apply_selection_tech=False
                   )

    del testset
    del test_loader

    # ==========================================================================
    #                               TRAIN SET
    # ==========================================================================
    # in case of ours:
    if args.al_type == constants.AL_LP:
        # performance over ALL pseudo-labeled samples. ========================
        new_samples = []

        for idtoadd in all_pairs.keys():
            stoadd = deepcopy(tr_original[ids_org.index(idtoadd)])
            stoadd[4] = constants.L  # convert to sup.
            new_samples.append(stoadd)
        # measure perf over pseudo-labeled samples using the true labels for
        # evaluation.
        if len(new_samples) > 0:
            # 1. evaluate the performance using standard functions.
            trainset_eval, train_eval_loader = get_validationset(
                args,
                new_samples,
                transform_tensor,
                padding_size_eval,
                batch_size=None
               )

            final_validate(model,
                           train_eval_loader,
                           CRITERION,
                           DEVICE,
                           trainset_eval,
                           OUTD_TR.folder,
                           args,
                           log_file=results_log,
                           name_set="trainset",
                           pseudo_labeled=True,
                           store_results=False,
                           apply_selection_tech=False
                           )

            # 2. pseudo-annotate all the pairs. ++++++++++++++++++++++++++++++++
            # warning: all_pairs contains all the pairs that will be used for
            # the next al round as pseudo-segmented samples.
            # however, some of these samples maybe queried to be labeled by
            # the oracle for the next al round. so, we need to remove them
            # BEFORE storing `previous_pairs`. and delete the stored masks.
            # we will do these actions after we determine the queries (See
            # line @FIXERROR).

            # threshold = args.seg_threshold
            if args.estimate_pseg_thres:
                # zout2 = estimate_best_seg_thres(model,
                #                                validset,
                #                                valid_loader,
                #                                CRITERION,
                #                                DEVICE,
                #                                args,
                #                                epoch=epoch,
                #                                cycle=cycle,
                #                                log_file=training_log
                #                                 )
                for zk in zout22.keys():
                    propagation['vl_final_' + zk] = zout22[zk]

                # threshold = zout22['best_threshold']

            propagation["pseudo_seg_thres_final"] = threshold
            log(results_log, "SEG-THRESHOLD_FINAL: {} [EST-VL: {}]".format(
                threshold, args.estimate_pseg_thres))

            outzk = pseudo_annotate_some_pairs(model,
                                               all_pairs,
                                               new_samples,
                                               args,
                                               CRITERION,
                                               transform_tensor,
                                               padding_size_eval,
                                               DEVICE,
                                               fd_p_msks,
                                               SHARED_OPT_MASKS,
                                               threshold,
                                               results_log,
                                               "SEG-PL-FINAL"
                                               )
            propagation["sum_dice_all_pairs"] = outzk['sum_dice']
            propagation["number_all_pairs"] = outzk['n_samples']

            del trainset_eval
            del train_eval_loader

    # validset maybe needed above...

    del validset
    del valid_loader

    # full sup data.
    if args.al_type == constants.AL_LP:
        train_samples = deepcopy(train_samples_before_merging)  # ful.sup.

    trainset_eval, train_eval_loader = get_validationset(args,
                                                         train_samples,
                                                         transform_tensor,
                                                         padding_size_eval,
                                                         batch_size=None
                                                         )

    final_validate(model,
                   train_eval_loader,
                   CRITERION,
                   DEVICE,
                   trainset_eval,
                   OUTD_TR.folder,
                   args,
                   log_file=results_log,
                   name_set="trainset",
                   store_results=False,
                   apply_selection_tech=False
                   )
    del trainset_eval
    del train_eval_loader

    # leftover =================================================================
    cnd = (args.al_type not in [constants.AL_FULL_SUP, constants.AL_WSL])
    cnd &= (tr_leftovers != [])

    if cnd:
        current_samples = deepcopy(tr_leftovers)
        for i in range(len(current_samples)):
            current_samples[i][4] = constants.L

        if len(current_samples) > 0:
            trainset_eval, train_eval_loader = get_validationset(
                args,
                current_samples,
                transform_tensor,
                padding_size_eval,
                batch_size=1
            )

            final_validate(model,
                           train_eval_loader,
                           CRITERION,
                           DEVICE,
                           trainset_eval,
                           OUTD_LO.folder,
                           args,
                           log_file=results_log,
                           name_set="trainset-leftovers",
                           store_results=False,
                           apply_selection_tech=True
                           )
            del trainset_eval
            del train_eval_loader

    # ==========================================================================
    #                 START: PREPARE DATA FOR THE NEXT ACTIVE LEARNING ROUND
    # ==========================================================================

    if args.al_type not in [constants.AL_FULL_SUP, constants.AL_WSL]:
        announce_msg("Preparing next active learning round: {}/{}...".format(
            args.al_it, args.al_type
        ))
        reset_seed(int(os.environ["MYSEED"]) + args.al_it)
        t0 = dt.datetime.now()
        tr_leftovers_effective = tr_leftovers

        if args.protect_pl and (args.al_type == constants.AL_LP):
            keys_pairs = list(all_pairs.keys())
            # protect only if the number of samples left  minus the pairs is
            # >= than the number of classes.

            cnd = (len(keys_pairs) > 0)
            remains = len(tr_leftovers) - len(keys_pairs)
            # recall that pairs_samples \subseteq tr_leftovers.
            cnd &= (remains >= len(list(args.name_classes.keys())))

            # this last condition is not perfect since it ignores the
            # unbalanced classes. we keep protecting/sampling as long as the
            # number of remained samples is >= than the number of
            # classes. we do not check if each class still has a sample to do
            # a balanced sampling.
            # if the number of remaining samples is less than the number of
            # classes, we allow sampling from pseudo-labeled samples.



            if cnd:
                announce_msg("PROTECTING PSEUDO-LABELED SAMPLES: "
                             "{} samples.".format(len(keys_pairs)))

                tr_leftovers_effective = [
                    elm for elm in tr_leftovers if elm[0] not in keys_pairs
                ]

        set_default_seed()
        tr_next_al = sampler(tr_samples=tr_leftovers_effective,
                             ids_current_tr=ids_curt,
                             simsfd=SIMS,
                             args=args,
                             exp_fd=OUTD
                             )
        set_default_seed()
        # save the next samples in COMMON
        base_f = 'train_{}.csv'.format(args.al_it + 1)
        al_outf = join(COMMON, base_f)
        csv_writer(clear_rootpath(tr_next_al, args),
                   al_outf
                   )
        # shutil.copyfile(al_outf, join(OUTD, base_f))
        announce_msg("For the next AL iteration, the oracle labeled {} "
                     "new samples.".format(len(tr_next_al)))
        msg = "Preparing next active learning round...end. {}".format(
            dt.datetime.now() - t0
        )
        announce_msg(msg)
        log(training_log, msg)


        #@FIXERROR -------------------------------------------------------------
        if args.al_type == constants.AL_LP:
            cur_full_sup = deepcopy(train_samples_before_merging)  # fully-sup
            cur_and_next_f_sup = tr_next_al + cur_full_sup
            for sm in cur_and_next_f_sup:
                id_sup = sm[0]
                if id_sup in all_pairs:
                    all_pairs.pop(id_sup)  # remove the sample... no longer
                    # useful as pseudo-labeled sample.

            # store all the current pairs. they will be used as
            # pseudo-labeled without stats. constraints in the next al round.
            path_tr_p = join(COMMON, 'train_pairs_{}.pkl'.format(args.al_it))
            with open(path_tr_p, 'wb') as fp:
                pkl.dump(all_pairs, fp, protocol=pkl.HIGHEST_PROTOCOL)

            msg = "Nbr-pseuod-labeled for next al " \
                  "round: {}.".format(len(list(all_pairs.keys())))
            print(msg)
            log(results_log, msg)
            # CLEANING...
            clean_shared_masks(args,
                               SHARED_OPT_MASKS,
                               metrics_fd,
                               all_pairs
                               )

        # ----------------------------------------------------------------------

        # stop active learning round if tr_next_al is []. send message
        #  to net rounds...

        if tr_next_al == []:
            with open(join(COMMON, 'stop_active_learning.txt'), 'w') as fend:
                fend.write("Stop active learning. there are no sample left to "
                           "select from.")

    # ==========================================================================
    #                 END: PREPARE DATA FOR THE NEXT ACTIVE LEARNING ROUND
    # ==========================================================================

    # ==========================================================================
    #                   PLOT ACTIVE LEARNING STATS
    # ==========================================================================
    if "CC_CLUSTER" not in os.environ.keys():
        # as a rule, we do not plot on servers with time limit such as cc
        # because we need to restart the code. when using nodes, restarting
        # the code will not be able to have all the previous al rounds in the
        # node leading to an error.
        # for the case of cc, we plot in scratch.
        plotter = PlotActiveLearningRounds(folder_rounds=ROUNDS,
                                           task=args.task,
                                           max_al_its=args.max_al_its
                                           )
        plotter()
    # ==========================================================================
    #                   PLOT ACTIVE LEARNING STATS
    # ==========================================================================

    # Move the state dict of the best model into CPU, then save it.
    best_state_dict_cpu = copy_model_state_dict_from_gpu_to_cpu(model)
    torch.save(best_state_dict_cpu, join(OUTD, "best_model.pt"))

    msg = "End final processing. ***DEBUG MODE OFF*** Time: {}".format(
        dt.datetime.now() - tx0)
    announce_msg(msg)
    log(training_log, msg)
    # use this at the beginning of this file to check that this round is done so
    # to not do it again.
    with open(join(OUTD, 'end.txt'), 'w') as fend:
        fend.write("Done.")

    # =========================== FINAL MOVE OF FILES =========================
    if "CC_CLUSTER" in os.environ.keys():
        print("Going to move things from node to scratch...")
        FOLDER = join(placement_scr, parent, tag)
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        # compress, and send to scratch.

        def create_cmds_folder(fd_to_move):
            """
            Create the command that moves a folder.
            """
            cmdx = [
                "cd {} ".format(join(placement_node, parent, tag)),
                "tar -cvf {}.tar.gz {} ".format(fd_to_move, fd_to_move),
                "cp {}.tar.gz {}".format(fd_to_move, FOLDER),
                "cd {} ".format(FOLDER),
                "tar -xvf  {}.tar.gz --overwrite ".format(fd_to_move),
                "rm {}.tar.gz ".format(fd_to_move)
                ]
            cmdx = " && ".join(cmdx)

            return cmdx

        fds_to_move = [exp_name, 'common', 'vision', 'shared_opt_masks']
        for fdx in fds_to_move:
            lcmd = create_cmds_folder(fdx)
            print("Running bash-cmds: \n{}".format(lcmd.replace("&& ", "\n")))
            subprocess.run(lcmd, shell=True, check=True)

    # TODO: delete bin_masks cc on other servers.

        # plot on scratch instead of node (cc only).
        plotter = PlotActiveLearningRounds(
            folder_rounds=join(placement_scr, parent, tag),
            task=args.task,
            max_al_its=args.max_al_its
        )
        plotter()


    # todo: track time properly in pkl file and for each segment: total,
    #  cycle0, cycle1.
    total_time = "AL_it: {} \nTotal running time: {}.".format(
        args.al_it, dt.datetime.now() - init_time)
    if "CC_CLUSTER" in os.environ.keys():
        # write in scratch not node.
        OUTDSCRATCH = join(placement_scr, parent, tag, exp_name)
        with open(join(OUTDSCRATCH, 'time.txt'), 'w') as fend:
            fend.write(total_time)
    else:
        # write in fdout of exp.
        with open(join(OUTD, 'time.txt'), 'w') as fend:
            fend.write(total_time)

    # clean shared_opt_masks on scratch (every time) +++++++++++++++++++++++++++
    # because the previous cleanings were done on the node.
    #
    cnd = (args.al_type == constants.AL_LP)
    cnd &= args.share_masks
    cnd &= ("CC_CLUSTER" in os.environ.keys())
    if cnd:
        sc_sh_op_msk = join(placement_scr, parent, tag, "shared_opt_masks")
        sc_metrics_fd = join(sc_sh_op_msk, "metrics")
        clean_shared_masks(args=args,
                           SHARED_OPT_MASKS=sc_sh_op_msk,
                           metrics_fd=sc_metrics_fd,
                           all_pairs=all_pairs
                           )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # compress `shared_opt_masks`. they may contain a lot of files (bad for
    # cc.). do it only at the last al round. delete the folder itself.

    cnd = (args.al_it == (args.max_al_its - 1))
    cnd &= args.share_masks
    cnd &= ("CC_CLUSTER" in os.environ.keys())
    if cnd:
        cmdx = [
            "cd {} ".format(join(placement_scr, parent, tag)),
            "tar -cvf shared_opt_masks.tar.gz shared_opt_masks",
            "rm -r shared_opt_masks"
        ]
        cmdx = " && ".join(cmdx)
        announce_msg("Final al round. cleaning `shared_opt_masks`.")
        print("Running bash-cmds: \n{}".format(cmdx.replace("&& ", "\n")))
        subprocess.run(cmdx, shell=True, check=True)

    announce_msg("*END*")
    # ==========================================================================
    #                              END
    # ==========================================================================
