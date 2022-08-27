import os
import datetime as dt
from os.path import join
import subprocess
import pickle as pkl
from copy import deepcopy


import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform


from tools import log
from tools import VisualsePredSegmentation

from deeplearning.criteria import Metrics, Entropy


from prologues import get_validationset
from prologues import clean_shared_masks
from prologues import seedj


from shared import announce_msg
from shared import CONST1
from shared import check_nans


from reproducibility import reset_seed
from reproducibility import set_default_seed


import constants


def build_target_cl(args,
                    labels,
                    tags,
                    dataset,
                    device
                    ):
    """
    Build targets for our method for task  task: CL.
    1. CL: create targets to do KL.
    """
    msg = 'expected task {} but found {}'.format(constants.CL, args.task)
    assert args.task == constants.CL, msg

    n = labels.shape[0]
    nbr_cl = len(list(dataset.name_classes.keys()))
    target = torch.zeros((n, nbr_cl), device=device, dtype=torch.float32,
                         requires_grad=False)
    # index of L, and LP tags.
    ind_l = (tags == constants.L).nonzero().squeeze(dim=1)  # labeled.
    ind_lp = (tags == constants.PL).nonzero().squeeze(dim=1)  # pseudo-labeled.
    if ind_l.numel() > 0:
        target[ind_l, labels[ind_l]] = 1.

    if ind_lp.numel() > 0:
        msg = "0< certainty <=100. found {}".format(args.certainty)  # deleted
        # certainty from args.

        assert 0. < args.certainty <= 100., msg
        certainty = args.certainty / 100.
        if certainty == 1.:
            target[ind_lp, labels[ind_lp]] = 1.
        else:
            leftover = (1. - certainty) / (nbr_cl - 1.)
            dist_unif = Uniform(torch.tensor([1e-10]), torch.tensor([leftover]))
            for i in range(ind_lp.numel()):
                idx = ind_lp[i]
                label = labels[idx]
                smooth = dist_unif.sample((nbr_cl, )).squeeze()
                smooth[label] = 0.
                target[idx, :] = smooth
                target[idx, label] = 1. - smooth.sum()
                # the above does not grantee that sum(row) is exactly one.

                # assert target[idx, :].sum() == 1., "expected sum to 1, " \
                #                                    "but found {}".format(
                #     target[idx, :].sum()
                # )
    return target


def train_one_epoch(model,
                    optimizer,
                    dataloader,
                    criterion,
                    device,
                    tr_stats,
                    args,
                    dataset,
                    epoch=0,
                    cycle=0,
                    log_file=None,
                    ALLOW_MULTIGPUS=False,
                    NBRGPUS=1
                    ):
    """
    Perform one epoch of training.
    :param model: instance of a model.
    :param optimizer: instance of an optimizer.
    :param dataloader: list of two instance of a dataloader: L u L`, U.
    :param criterion: instance of a learning criterion.
    :param device: a device.
    :param tr_stats: dict that holds the states of the training. or
    None.
    :param args: args of the main.py
    :param dataset: the dataset.
    :param epoch: int, the current epoch.
    :param cycle: int, the current cylce.
    :param log_file: a logfile.
    :param ALLOW_MULTIGPUS: bool. If True, we are in multiGPU mode.
    :param NBRGPUS: int, number of GPUs.
    :return:
    """
    reset_seed(seedj(epoch, 3, cycle, CONST1))

    model.train()
    criterion.train()
    metrics = Metrics(threshold=args.seg_threshold).to(device)
    metrics.eval()

    length = len(dataloader)
    t0 = dt.datetime.now()
    # to track stats.
    keys = ["acc", "dice_idx"]
    losses_kz = ["total_loss", "cl_loss", "seg_l_loss", "seg_lp_loss"]
    keys = keys + losses_kz
    tracker = dict()
    for k in keys:
        tracker[k] = 0.

    n_samples = 0.
    n_sam_dice = 0.
    # ignoring samples is on only: 1. cam16 dataset. 2. wsl method.
    # for the rest of the methods, normal samples are completely dropped before
    # starting training.
    cnd_dice = (args.dataset == constants.CAM16)
    cnd_dice &= (args.al_type == constants.AL_WSL)

    for i, (ids, data, mask, label, tag, crop_cord) in tqdm.tqdm(
            enumerate(dataloader), ncols=80, total=length):
        seedx = int(os.environ["MYSEED"]) + epoch + (cycle + 1) * 10
        reset_seed(seedx)

        targets = None
        masks_pred, masks_trg = None, None
        if mask is not None:
            masks_trg = mask.to(device)

        if (args.al_type == constants.AL_LP) and (args.task == constants.CL):
            targets = build_target_cl(args=args, labels=label, tags=tag,
                                      dataset=dataset, device=device)
            reset_seed(seedx)

        seedx += i

        data = data.to(device)
        labels = label.to(device)

        if targets is not None:
            targets = targets.to(device)

        bsz = data.size()[0]

        model.zero_grad()
        prngs_cuda = None

        # Optimization:
        if not ALLOW_MULTIGPUS:
            if "CC_CLUSTER" in os.environ.keys():
                msg = "Something wrong. You deactivated multigpu mode, " \
                      "but we find {} GPUs. This will not guarantee " \
                      "reproducibility. We do not know why you did that. " \
                      "Exiting ... [NOT OK]".format(NBRGPUS)
                assert NBRGPUS <= 1, msg
            seeds_threads = None
        else:
            msg = "Something is wrong. You asked for multigpu mode. But, " \
                  "we found {} GPUs. Exiting .... [NOT OK]".format(NBRGPUS)
            assert NBRGPUS > 1, msg
            # The seeds are generated randomly before calling the threads.
            reset_seed(seedx)
            seeds_threads = torch.randint(
                0, np.iinfo(np.uint32).max + 1, (NBRGPUS, )).to(device)
            reset_seed(seedx)
            prngs_cuda = []
            # Create different prng states of cuda before forking.
            for seed in seeds_threads:
                # get the corresponding state of the cuda prng with respect to
                # the seed.
                inter_seed = seed.cpu().item()
                # change the internal state of the prng to a random one using
                # the random seed so to capture it.
                torch.manual_seed(inter_seed)
                torch.cuda.manual_seed(inter_seed)
                # capture the prng state.
                prngs_cuda.append(torch.cuda.get_rng_state())
            reset_seed(seedx)

        if prngs_cuda is not None and prngs_cuda != []:
            prngs_cuda = torch.stack(prngs_cuda)

        reset_seed(seedx)
        scores, masks_pred, maps = model(x=data,
                                         seed=seeds_threads,
                                         prngs_cuda=prngs_cuda
                                        )
        reset_seed(seedx)

        # change to the predicted mask to be the maps if WSL.
        if args.al_type == constants.AL_WSL:
            lb_ = labels.view(-1, 1, 1, 1)
            masks_pred = (maps.argmax(dim=1, keepdim=True) == lb_).float()
            # masks_pred contains binary values with float format.
            # shape (bsz, 1, h, w)
        else:
            # TODO: change normalization location. (future)
            masks_pred = torch.sigmoid(masks_pred)  # binary segmentation.


        losses = criterion(scores=scores,
                           labels=labels,
                           targets=targets,
                           masks_pred=masks_pred.view(bsz, -1),
                           masks_trg=masks_trg.view(bsz, -1),
                           tags=tag,
                           weights=None,
                           avg=True
                           )
        reset_seed(seedx)

        losses[0].backward()  # total loss.
        reset_seed(seedx)
        # Update params.
        optimizer.step()
        reset_seed(seedx)
        # End optimization.

        # metrics

        ignore_dice = None
        if cnd_dice:
            # ignore samples with label 'normal' (0) when computing dice.
            ignore_dice = (labels == 0).float().view(-1)


        metrx = metrics(scores=scores,
                        labels=labels,
                        tr_loss=criterion,
                        masks_pred=masks_pred.view(bsz, -1),
                        masks_trg=masks_trg.view(bsz, -1),
                        avg=False,
                        ignore_dice=ignore_dice
                        )

        # Update the tracker.
        tracker["acc"] += metrx[0].item()
        tracker["dice_idx"] += metrx[1].item()

        for j, los in enumerate(losses):
            tracker[losses_kz[j]] += los.item()

        n_samples += bsz
        if cnd_dice:
            n_sam_dice += ignore_dice.numel() - ignore_dice.sum()
        else:
            n_sam_dice += bsz

        # clear the memory
        del scores
        del masks_pred
        del maps

    # average.
    for k in keys:
        if k in losses_kz:
            tracker[k] /= float(length)
        elif (k == 'dice_idx') and cnd_dice:
            announce_msg("Ignore normal samples mode. "
                         "Total samples left: {}/Total: {}.".format(
                n_sam_dice, n_samples))
            if n_sam_dice == 0:
                tracker[k] = 0.
            else:
                tracker[k] /= float(n_sam_dice)
        else:
            tracker[k] /= float(n_samples)

    to_write = "Tr.Ep {:>2d}-{:>2d}: ACC: {:.2f}%, DICE: {:.2f}%, LR: {}, " \
               "time:{}".format(
        cycle,
        epoch,
        tracker["acc"] * 100.,
        tracker["dice_idx"] * 100.,
        ['{:.2e}'.format(group["lr"]) for group in optimizer.param_groups],
        dt.datetime.now() - t0
    )
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # Update stats:
    if tr_stats is not None:
        for k in keys:
            tr_stats[k].append(tracker[k])
    else:
        # convert each element into a list
        for k in keys:
            tracker[k] = [tracker[k]]
        tr_stats = deepcopy(tracker)

    reset_seed(seedj(epoch, 4, cycle, CONST1))

    return tr_stats


def validate(model,
             dataset,
             dataloader,
             criterion,
             device,
             stats,
             args,
             epoch=0,
             cycle=0,
             log_file=None
             ):
    """
    Perform a validation over a set.
    Dataset passed here must be in `validation` mode.
    Task: SEG.
    """
    set_default_seed()

    model.eval()
    criterion.eval()
    metrics = Metrics(threshold=args.seg_threshold).to(device)
    metrics.eval()

    length = len(dataloader)
    # to track stats.
    keys = ["acc", "dice_idx"]
    losses_kz = ["total_loss", "cl_loss", "seg_l_loss", "seg_lp_loss"]
    keys = keys + losses_kz
    tracker = dict()
    for k in keys:
        tracker[k] = 0.

    n_samples = 0.
    n_sam_dice = 0.

    # ignoring samples is on only: 1. cam16 dataset. 2. wsl method.
    # for the rest of the methods, normal samples are completely dropped before
    # starting training.
    cnd_dice = (args.dataset == constants.CAM16)
    cnd_dice &= (args.al_type == constants.AL_WSL)

    t0 = dt.datetime.now()

    with torch.no_grad():
        for i, (ids, data, mask, label, tag, crop_cord) in tqdm.tqdm(
                enumerate(dataloader), ncols=80, total=length):

            reset_seed(int(os.environ["MYSEED"]))

            targets = None  # not needed for SEG task.
            masks_trg = None
            bsz = data.size()[0]

            data = data.to(device)
            labels = label.to(device)
            if mask is not None:
                masks_trg = mask.to(device)

            scores, masks_pred, maps = model(x=data, seed=None)

            check_nans(maps, "fresh-maps")

            # change to the predicted mask to be the maps if WSL.
            if args.al_type == constants.AL_WSL:
                lb_ = labels.view(-1, 1, 1, 1)
                masks_pred = (maps.argmax(dim=1, keepdim=True) == lb_).float()


            # resize pred mask if sizes mismatch.
            if masks_pred.shape != masks_trg.shape:
                _, _, oh, ow = masks_trg.shape
                masks_pred = dataset.turnback_mask_tensor_into_original_size(
                    pred_masks=masks_pred, oh=oh, ow=ow
                )
                check_nans(masks_pred, "mask-pred-back-to-normal-size")

            if args.al_type != constants.AL_WSL:
                # TODO: change normalization location. (future)
                masks_pred = torch.sigmoid(masks_pred)  # binary segmentation

            losses = criterion(scores=scores,
                               labels=labels,
                               targets=targets,
                               masks_pred=masks_pred.view(bsz, -1),
                               masks_trg=masks_trg.view(bsz, -1),
                               tags=tag,
                               weights=None,
                               avg=False
                               )

            # metrics
            ignore_dice = None
            if cnd_dice:
                # ignore samples with label 'normal' (0) when computing dice.
                ignore_dice = (labels == 0).float().view(-1)

            metrx = metrics(scores=scores,
                            labels=labels,
                            tr_loss=criterion,
                            masks_pred=masks_pred.view(bsz, -1),
                            masks_trg=masks_trg.view(bsz, -1),
                            avg=False,
                            ignore_dice=ignore_dice
                            )

            tracker["acc"] += metrx[0].item()
            tracker["dice_idx"] += metrx[1].item()

            # print("Dice valid i: {} {}. datashape {} id {}".format(
            #     i, metrx[1].item() * 100., data.shape, ids))


            for j, los in enumerate(losses):
                tracker[losses_kz[j]] += los.item()

            n_samples += bsz  # nbr samples.

            if cnd_dice:
                n_sam_dice += ignore_dice.numel() - ignore_dice.sum()
            else:
                n_sam_dice += bsz

            # clear the memory
            del scores
            del masks_pred
            del maps

    # average.
    for k in keys:
        if (k == 'dice_idx') and cnd_dice:
            announce_msg("Ignore normal samples mode. "
                         "Total samples left: {}/Total: {}.".format(
                n_sam_dice, n_samples))
            if n_sam_dice == 0:
                tracker[k] = 0.
            else:
                tracker[k] /= float(n_sam_dice)
        else:
            tracker[k] /= float(n_samples)

    to_write = "VL {:>2d}-{:>2d}: ACC: {:.4f}, DICE: {:.4f}, time:{}".format(
                cycle, epoch, tracker["acc"] * 100., tracker["dice_idx"] * 100.,
                dt.datetime.now() - t0
                 )
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # Update stats.
    if stats is not None:
        for k in keys:
            stats[k].append(tracker[k])
    else:
        # convert each element into a list
        for k in keys:
            tracker[k] = [tracker[k]]
        stats = deepcopy(tracker)

    set_default_seed()

    return stats


def estimate_best_seg_thres(model,
                            dataset,
                            dataloader,
                            criterion,
                            device,
                            args,
                            epoch=0,
                            cycle=0,
                            log_file=None
                            ):
    """
    Perform a validation over a set.
    Allows estimating a segmentation threshold based on this evaluation.

    This is intended fo the `validationset` where we can estimate the best
    threshold since the samples are pixel-wise labeled.

    We specify a set of theresholds, and pick the one that has the best mean
    IOU score. MIOU is better then Dice index.

    Dataset passed here must be in `validation` mode.
    Task: SEG.
    """
    msg = "Can't use/estimate threshold over AL_WSL. masks are already binary."
    assert args.al_type != constants.AL_WSL, msg

    set_default_seed()

    announce_msg("Estimating threshold on validation set.")
    set_default_seed()

    model.eval()
    # the specified threshold here does not matter. we will use a different
    # ones later.
    metrics = Metrics(threshold=args.seg_threshold).to(device)
    metrics.eval()

    length = len(dataloader)
    l_thress = np.arange(start=0.05, stop=1., step=0.01)
    # to track stats.
    avg_dice = np.zeros(l_thress.shape)
    avg_miou = np.zeros(l_thress.shape)

    nbr_ths = l_thress.size

    n_samples = 0.
    n_sam_dice = 0.
    # ignoring samples is on only: 1. cam16 dataset. 2. wsl method.
    # for the rest of the methods, normal samples are completely dropped before
    # starting training.
    cnd_dice = (args.dataset == constants.CAM16)
    cnd_dice &= (args.al_type == constants.AL_WSL)

    t0 = dt.datetime.now()

    with torch.no_grad():
        for i, (ids, data, mask, label, tag, crop_cord) in tqdm.tqdm(
                enumerate(dataloader), ncols=80, total=length):

            reset_seed(int(os.environ["MYSEED"]))

            targets = None  # not needed for SEG task.
            masks_trg = None
            bsz = data.size()[0]
            # assert bsz == 1, "batch size must be 1. found {}. ".format(bsz)

            data = data.to(device)
            labels = label.to(device)
            if mask is not None:
                masks_trg = mask.to(device)

            scores, masks_pred, maps = model(x=data, seed=None)

            # resize pred mask if sizes mismatch.
            if masks_pred.shape != masks_trg.shape:
                _, _, oh, ow = masks_trg.shape
                masks_pred = dataset.turnback_mask_tensor_into_original_size(
                    pred_masks=masks_pred, oh=oh, ow=ow
                )
            # TODO: change normalization location. (future)
            masks_pred = torch.sigmoid(masks_pred)

            ignore_dice = None
            if cnd_dice:
                # ignore samples with label 'normal' (0) when computing dice.
                ignore_dice = (labels == 0).float().view(-1)

            for ii, tt in enumerate(l_thress):
                metrx = metrics(scores=scores,
                                labels=labels,
                                tr_loss=criterion,
                                masks_pred=masks_pred.view(bsz, -1),
                                masks_trg=masks_trg.view(bsz, -1),
                                avg=False,
                                threshold=tt,
                                ignore_dice=ignore_dice
                                )

                avg_dice[ii] += metrx[1].item()
                avg_miou[ii] += metrx[2].item()

            n_samples += bsz  # nbr samples.

            if cnd_dice:
                n_sam_dice += ignore_dice.numel() - ignore_dice.sum()
            else:
                n_sam_dice += bsz

            # clear the memory
            del scores
            del masks_pred

    idx = avg_miou.argmax()
    best_threshold = l_thress[idx]
    if cnd_dice and (n_sam_dice != 0):
        best_dice = avg_dice[idx] / float(n_sam_dice)
        best_miou = avg_miou[idx] / float(n_sam_dice)
    else:
        best_dice = avg_dice[idx] / float(n_samples)
        best_miou = avg_miou[idx] / float(n_samples)

    out = {
        "best_threshold": best_threshold,
        "best_dice": best_dice,
        "best_miou": best_miou
    }

    to_write = "VL [ESTIM-THRESH] {:>2d}-{:>2d}. BEST-DICE: {:.2f}, " \
               "BEST-MIOU: {:.2f}. " \
               "time:{}".format(cycle,
                                epoch,
                                best_dice * 100.,
                                best_miou * 100.,
                                dt.datetime.now() - t0
    )
    print(to_write)
    if log_file:
        log(log_file, to_write)

    to_write =  "best threshold: {:.3f}. " \
                 "BEST-DICE-VL: {:.2f}%, " \
                 "BEST-MIOU-VL: {:.2f}%.".format(
        best_threshold, best_dice * 100., best_miou * 100.
    )
    announce_msg(to_write)
    if log_file:
        log(log_file, to_write)

    set_default_seed()

    return out


def pseudo_annotate_some_pairs(model,
                               some_pairs,
                               samples,
                               args,
                               criterion,
                               transform_tensor,
                               padding_size_eval,
                               device,
                               fd_p_msks,
                               SHARED_OPT_MASKS,
                               threshold=0.5,
                               results_log=None,
                               txt=""
                               ):
    """
    Pseudo-annotate some pairs.
    :param some_pairs:
    :param samples:
    :param threshold:
    :return:
    """
    set_default_seed()

    metrics_fd = join(fd_p_msks, "metrics")
    # create dataset.
    evalset, eval_loader  = get_validationset(args,
                                              samples,
                                              transform_tensor,
                                              padding_size_eval,
                                              batch_size=1
                                              )

    set_default_seed()
    sum_dice, n_samples = _pseudo_annotate(model=model,
                                           dataset=evalset,
                                           dataloader=eval_loader,
                                           criterion=criterion,
                                           device=device,
                                           pairs=some_pairs,
                                           metrics_fd=metrics_fd,
                                           fd_p_msks=fd_p_msks,
                                           args=args,
                                           SHARED_OPT_MASKS=SHARED_OPT_MASKS,
                                           threshold=threshold
                                           )



    if results_log is not None:
        avg_dice = (sum_dice / float(n_samples)) if n_samples != 0 else 0.
        msg = "{}: {} [scale_seg_u: {}] ".format(
            txt,
            avg_dice * 100.,
            args.scale_seg_u
            )
        log(results_log, msg)
        print(msg)


    out = {
        "sum_dice": sum_dice,
        "n_samples": n_samples
    }

    return deepcopy(out)


def _pseudo_annotate(model,
                     dataset,
                     dataloader,
                     criterion,
                     device,
                     pairs,
                     metrics_fd,
                     fd_p_msks,
                     args,
                     SHARED_OPT_MASKS,
                     threshold=0.5
                     ):
    """
    Perform a validation over a set.
    Allows storing pseudo-masks.


    Dataset passed here must be in `validation` mode.
    Task: SEG.
    """
    set_default_seed()

    model.eval()
    metrics = Metrics(threshold=threshold).to(device)
    metrics.eval()

    length = len(dataloader)
    # to track stats.
    sum_dice = 0.
    n_samples = 0.

    with torch.no_grad():
        for i, (ids, data, mask, label, tag, crop_cord) in tqdm.tqdm(
                enumerate(dataloader), ncols=80, total=length):

            reset_seed(int(os.environ["MYSEED"]))

            targets = None  # not needed for SEG task.
            masks_trg = None
            bsz = data.size()[0]
            assert bsz == 1, "batch size must be 1. found {}. ".format(bsz)

            id_u = ids[0]
            id_l = pairs[id_u]
            name_file = "{}-{}".format(id_u, id_l)
            path_to_save_mask = join(fd_p_msks, "{}.bmp".format(name_file))
            path_to_save_metric = join(metrics_fd, "{}.pkl".format(name_file))

            data = data.to(device)
            labels = label.to(device)
            if mask is not None:
                masks_trg = mask.to(device)

            scores, masks_pred, maps = model(x=data, seed=None)

            # resize pred mask if sizes mismatch.
            if masks_pred.shape != masks_trg.shape:
                _, _, oh, ow = masks_trg.shape
                masks_pred = dataset.turnback_mask_tensor_into_original_size(
                    pred_masks=masks_pred, oh=oh, ow=ow
                )
            # TODO: change normalization location. (future)
            masks_pred = torch.sigmoid(masks_pred)

            metrx = metrics(scores=scores,
                            labels=labels,
                            tr_loss=criterion,
                            masks_pred=masks_pred.view(bsz, -1),
                            masks_trg=masks_trg.view(bsz, -1),
                            avg=False
                            )

            sum_dice += metrx[1].item()

            n_samples += bsz  # nbr samples.

            # store the mask and metric. =======================================
            bin_mask = metrics.get_binary_mask(masks_pred, threshold=threshold)
            bin_mask = bin_mask.detach().cpu().squeeze().numpy()
            # issue with mode=1...
            # https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-
            # from-array-with-mode-1
            img_mask = Image.fromarray(bin_mask.astype(np.uint8) * 255,
                                       mode='L').convert('1')

            img_mask.save(path_to_save_mask)
            with open(path_to_save_metric, "wb") as fout:
                pkl.dump(
                    {"dice_u": metrx[1].item()}, fout,
                    protocol=pkl.HIGHEST_PROTOCOL)

            # ==================================================================


            # clear the memory
            del scores
            del masks_pred

    # clean: remove duplicates.
    clean_shared_masks(args=args,
                       SHARED_OPT_MASKS=SHARED_OPT_MASKS,
                       metrics_fd=metrics_fd,
                       all_pairs=pairs
                       )

    set_default_seed()

    return sum_dice, n_samples


def final_validate(model,
                   dataloader,
                   criterion,
                   device,
                   dataset,
                   outd,
                   args,
                   log_file=None,
                   name_set="",
                   pseudo_labeled=False,
                   store_results=False,
                   apply_selection_tech=False
                   ):
    """
    Perform a final evaluation of a set.
    (images do not have the same size, so we can't stack them in one tensor).
    Validation samples may be large to fit all in the GPU at once.

    This is similar to validate() but it can operate on all types of
    datasets (train, valid, test). It selects only L. Also, it does some other
    operations related to drawing and final training tasks.

    :param outd: str, output directory of this dataset.
    :param name_set: str, name to indicate which set is being processed. e.g.:
           trainset, validset, testset.
    :param pseudo_labeled: bool. if true, the dataloader is loading the set
           of samples that has been pseudo-labeled. this is the case only for
           our method. if false, the samples are fully labeled. this allows to
          measure the performance over the pseudo-labeled samples separately.
    :param store_results: bool (default: False). if true, results (
           predictions) are stored. Useful for test set to store predictions.
           can be disabled for train/valid sets.
    :param apply_selection_tech: bool. if true, we compute stats that will be
           used for sample selection for the nex AL round. useful on the
           leftovers.
    """
    set_default_seed()

    outd_data = join(outd, "prediction")
    if not os.path.exists(outd_data):
        os.makedirs(outd_data)

    # TODO: rescale depending on the dataset.
    visualisor = VisualsePredSegmentation(height_tag=args.height_tag,
                                          show_tags=True,
                                          scale=0.5
                                          )

    model.eval()
    criterion.eval()
    metrics = Metrics(threshold=args.seg_threshold).to(device)
    metrics.eval()

    length = len(dataloader)
    # to track stats.
    keys_perf = ["acc", "dice_idx"]
    losses_kz = ["total_loss", "cl_loss", "seg_l_loss", "seg_lp_loss"]
    per_sample_kz = ["pred_labels", "pred_masks", "ids"]
    keys = keys_perf + losses_kz + per_sample_kz
    tracker = dict()
    for k in keys:
        if k not in per_sample_kz:
            tracker[k] = 0.
        else:
            tracker[k] = []

    n_samples = 0.
    n_sam_dice = 0.
    # ignoring samples is on only: 1. cam16 dataset. 2. wsl method.
    # for the rest of the methods, normal samples are completely dropped before
    # starting training.
    cnd_dice = (args.dataset == constants.CAM16)
    cnd_dice &= (args.al_type == constants.AL_WSL)

    # Selection criteria.
    cp_entropy = Entropy()  # to copmute entropy.
    entropy = []
    mcdropout_var = []

    t0 = dt.datetime.now()

    # conditioning
    entropy_cnd = (args.al_type == constants.AL_ENTROPY)
    entropy_cnd = entropy_cnd or ((args.al_type == constants.AL_LP) and (
            args.clustering == constants.CLUSTER_ENTROPY))
    mcdropout_cnd = (args.al_type == constants.AL_MCDROPOUT)

    # cnd_asrt = (args.task == constants.SEG)
    cnd_asrt = store_results
    cnd_asrt |= apply_selection_tech

    with torch.no_grad():
        for i, (ids, data, mask, label, tag, crop_cord) in tqdm.tqdm(
                enumerate(dataloader), ncols=80, total=length):

            reset_seed(int(os.environ["MYSEED"]))

            targets = None
            bsz = data.size()[0]

            if cnd_asrt:
                msg = "batchsize must be 1 for segmentation task. " \
                      "found {}.".format(bsz)
                assert bsz == 1, msg

            data = data.to(device)
            labels = label.to(device)
            masks_trg = mask.to(device)

            scores, masks_pred, maps = model(x=data, seed=None)

            # change to the predicted mask to be the maps if WSL.
            if args.al_type == constants.AL_WSL:
                lb_ = labels.view(-1, 1, 1, 1)
                masks_pred = (maps.argmax(dim=1, keepdim=True) == lb_).float()

            # resize pred mask if sizes mismatch.
            if masks_pred.shape != masks_trg.shape:
                _, _, oh, ow = masks_trg.shape
                masks_pred = dataset.turnback_mask_tensor_into_original_size(
                    pred_masks=masks_pred,
                    oh=oh,
                    ow=ow
                )

            if args.al_type != constants.AL_WSL:
                # TODO: change normalization location. (future)
                masks_pred = torch.sigmoid(masks_pred)  # binary segmentation

            losses_ = criterion(scores=scores,
                                labels=labels,
                                targets=targets,
                                masks_pred=masks_pred.view(bsz, -1),
                                masks_trg=masks_trg.view(bsz, -1),
                                tags=tag,
                                weights=None,
                                avg=False
                                )

            # metrics
            ignore_dice = None
            if cnd_dice:
                # ignore samples with label 'normal' (0) when computing dice.
                ignore_dice = (labels == 0).float().view(-1)

            metrx = metrics(scores=scores,
                            labels=labels,
                            tr_loss=criterion,
                            masks_pred=masks_pred.view(bsz, -1),
                            masks_trg=masks_trg.view(bsz, -1),
                            avg=False,
                            ignore_dice=ignore_dice
                            )
            tracker["acc"] += metrx[0].item()
            tracker["dice_idx"] += metrx[1].item()

            for j, los in enumerate(losses_):
                tracker[losses_kz[j]] += los.item()

            # stored things
            # always store ids.
            tracker["ids"].extend(ids)

            if store_results:
                tracker["pred_labels"].extend(
                    scores.argmax(dim=1, keepdim=False).cpu().numpy().tolist()
                )
                # store binary masks
                tracker["pred_masks"].extend(
                    [(metrics.get_binary_mask(
                        masks_pred[kk]).detach().cpu().numpy() > 0.) for kk in
                     range(bsz)]
                )

            n_samples += bsz
            if cnd_dice:
                n_sam_dice += ignore_dice.numel() - ignore_dice.sum()
            else:
                n_sam_dice += bsz

            # ==================================================================
            #                    START: COMPUTE INFO. FOR SELECTION CRITERIA.
            # ==================================================================
            # 1. Entropy
            if entropy_cnd and apply_selection_tech:
                if args.task == constants.CL:
                    entropy.extend(
                        cp_entropy(
                            F.softmax(scores, dim=1)).cpu().numpy().tolist()
                    )
                elif args.task == constants.SEG:
                    assert bsz == 1, "batchsize must be 1. " \
                                     "found {}.".format(bsz)
                    # nbr_pixels, 2 (forg, backg).
                    pixel_probs = torch.cat((masks_pred.view(-1, 1),
                                             masks_pred.view(-1, 1)), dim=1)
                    avg_pixel_entropy = cp_entropy(pixel_probs).mean()
                    entropy.append(avg_pixel_entropy.cpu().item())
                else:
                    raise ValueError("Unknown task {}.".format(args.task))

            # 1. MC-Dropout
            if mcdropout_cnd and apply_selection_tech:

                reset_seed(int(os.environ["MYSEED"]))

                if args.task == constants.CL:
                    # todo
                    raise NotImplementedError
                elif args.task == constants.SEG:
                    assert bsz == 1, "batchsize must be 1. " \
                                     "found {}.".format(bsz)
                    stacked_masks = None
                    # turn on dropout
                    model.set_dropout_to_train_mode()
                    for it_mc in range(args.mcdropout_t):
                        scores, masks_pred, maps = model(x=data, seed=None)

                        # resize pred mask if sizes mismatch.
                        if masks_pred.shape != masks_trg.shape:
                            _, _, oh, ow = masks_trg.shape
                            masks_pred = \
                                dataset.turnback_mask_tensor_into_original_size(
                                    pred_masks=masks_pred, oh=oh, ow=ow
                                )

                        # TODO: change normalization location. (future)
                        masks_pred = torch.sigmoid(masks_pred)
                        # stack flatten masks horizontally.
                        if stacked_masks is None:
                            stacked_masks = masks_pred.view(-1, 1)
                        else:
                            stacked_masks = torch.cat(
                                (stacked_masks, masks_pred.view(-1, 1)),
                                dim=0
                            )
                    # compute variance per pixel, then average over the
                    # image. images have different sizes. if it is not the
                    # case, it is fine since we divide all sum_var with a
                    # constant.
                    variance = stacked_masks.var(
                        dim=0, unbiased=True).mean()
                    mcdropout_var.append(variance.cpu().item())

                    # turn off dropout
                    model.set_dropout_to_eval_mode()
                else:
                    raise ValueError(
                        "Unknown task {}.".format(args.task))
            # ==================================================================
            #                    END: COMPUTE INFO. FOR SELECTION CRITERIA.
            # ==================================================================

            # Visualize
            cnd = (args.task == constants.SEG)
            cnd &= (args.subtask == constants.SUBCLSEG)
            cnd &= store_results

            # todo: separate storing stats from plotting figure + storing it.
            # todo: add plot_figure var.
            if cnd:
                output_file = join(outd_data, "{}.jpeg".format(ids[0]))

                visualisor(
                    img_in=dataset.get_original_input_img(i),
                    mask_pred=metrics.get_binary_mask(
                        masks_pred).detach().cpu().squeeze().numpy(),
                    true_label=dataset.get_original_input_label_int(i),
                    label_pred=scores.argmax(dim=1, keepdim=False).cpu().numpy(
                    ).tolist()[0],
                    id_sample=ids[0],
                    name_classes=dataset.name_classes,
                    true_mask=masks_trg.cpu().squeeze().numpy(),
                    dice=metrx[1].item(),
                    output_file=output_file,
                    scale=None,
                    binarize_pred_mask=False,
                    cont_pred_msk=masks_pred.detach().cpu().squeeze().numpy()
                )

            # clear memory
            del scores
            del masks_pred
            del maps

    # avg: acc, dice_idx
    for k in keys_perf:
        if (k == 'dice_idx') and cnd_dice:
            announce_msg("Ignore normal samples mode. "
                         "Total samples left: {}/Total: {}.".format(
                n_sam_dice, n_samples))
            if n_sam_dice != 0:
                tracker[k] /= float(n_sam_dice)
            else:
                tracker[k] = 0.

        else:
            tracker[k] /= float(n_samples)

    # compress, then delete files to prevent overloading the disc quota of
    # number of files.
    def compress_del(src):
        """
        Compress a folder with name 'src' into 'src.zip' or 'src.tar.gz'.
        Then, delete the folder 'src'.
        :param src: str, absolute path to the folder to compress.
        :return:
        """
        try:
            cmd_compress = 'zip -rjq {}.zip {}'.format(src, src)
            print("Run: `{}`".format(cmd_compress))
            subprocess.run(cmd_compress, shell=True, check=True)
        except subprocess.CalledProcessError:
            cmd_compress = 'tar -zcf {}.tar.gz -C {} .'.format(src, src)
            print("Run: `{}`".format(cmd_compress))
            subprocess.run(cmd_compress, shell=True, check=True)

        cmd_del = 'rm -r {}'.format(src)
        print("Run: `{}`".format(cmd_del))
        os.system(cmd_del)

    compress_del(outd_data)

    to_write = "EVAL.FINAL {} -- pseudo-labeled {}: ACC: {:.3f}%," \
               " DICE: {:.3f}%, time:{}".format(
                name_set, pseudo_labeled, tracker["acc"] * 100.,
                tracker["dice_idx"] * 100., dt.datetime.now() - t0
               )
    to_write = "{} \n{} \n{}".format(10 * "=", to_write, 10 * "=")
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # store the stats in pickle.
    final_stats = dict()  # for a quick access: perf.
    for k in keys_perf:
        final_stats[k] = tracker[k] * 100.

    pred_stats = dict()  # it is heavy.
    for k in losses_kz + per_sample_kz:
        pred_stats[k] = tracker[k]

    pred_stats["pseudo-labeled"] = pseudo_labeled

    if pseudo_labeled:
        outfile_tracker = join(outd, 'final-tracker-{}-pseudoL-{}.pkl'.format(
            name_set, pseudo_labeled))
        outfile_pred = join(outd, 'final-pred-{}-pseudoL-{}.pkl'.format(
            name_set, pseudo_labeled))
    else:
        outfile_tracker = join(outd, 'final-tracker-{}.pkl'.format(name_set))
        outfile_pred = join(outd, 'final-pred-{}.pkl'.format(name_set))

    with open(outfile_tracker, 'wb') as fout:
        pkl.dump(final_stats, fout, protocol=pkl.HIGHEST_PROTOCOL)
    with open(outfile_pred, 'wb') as fout:
        pkl.dump(pred_stats, fout, protocol=pkl.HIGHEST_PROTOCOL)

    # store info. for selection techniques.
    # 1. Entropy.
    if entropy_cnd and apply_selection_tech:
        with open(join(outd, 'entropy-{}.pkl'.format(name_set)), 'wb') as fout:
            pkl.dump({'entropy': entropy,
                      'ids': tracker["ids"]}, fout,
                     protocol=pkl.HIGHEST_PROTOCOL)

    # 2. MC-Dropout.
    if mcdropout_cnd and apply_selection_tech:
        with open(join(outd, 'mc-dropout-{}.pkl'.format(name_set)),
                  'wb') as fout:
            pkl.dump({'mc-dropout-var': mcdropout_var,
                      'ids': tracker["ids"]}, fout,
                     protocol=pkl.HIGHEST_PROTOCOL)

    set_default_seed()
