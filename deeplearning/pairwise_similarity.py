import sys
import os
from os.path import join
import copy


from torch.utils.data import DataLoader
import torch
import tqdm
import pickle as pkl
import numpy as np
from torchvision import transforms

sys.path.append("..")
from reproducibility import set_default_seed
from loader import PhotoDataset, _init_fn, default_collate
from tools import get_transforms_tensor
from shared import announce_msg
import constants
from deeplearning.optimize_mask import SoftHistogram
from deeplearning.criteria import LossHistogramsMatching
from deeplearning.optimize_mask import GaussianSmoothing


__all__ = ["PairwiseSimilarity"]


class _ComputeSim(torch.nn.Module):
    """
    Torch class that computes the similarity between a vector and a set of
    vectors. use GPU for fast computation.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_ComputeSim, self).__init__()

    def forward(self, x1, x2):
        """
        Compute L2^2 distance between the vector x1 and the matrix x2 where
        each row of x2 is a vector with same dim as x1.

        :param x1: torch vector of dim (1, dim).
        :param x2: torch vector of dim (n, dim) where n is the number of
        vectors.

        :return : vector of dim (n) where
        each element i is ||x1 - x2[i, :]||_2^2.
        """
        assert x1.ndim == 2, "x1.ndim must be 2, but found {}.".format(x1.ndim)
        assert x1.size()[0] == 1, "x1.size[0] must be 1, but found {}.".format(
            x1.size()[0])
        assert x2.ndim == 2, "x2.ndim must be 2, but found {}.".format(x2.ndim)
        assert x1.size()[1] == x2.size()[1], "x1 dim={} and x2 dim={}. they " \
                                             "are supposed to be " \
                                             "similar.".format(x1.size()[1],
                                                               x2.size()[1])
        n = x2.size()[0]
        x = x1.repeat(n, 1)
        diff = x - x2

        return (diff * diff).sum(dim=1).squeeze()


class PairwiseSimilarity(object):
    """
    Compute the pairwise similarity between all the samples. Store the
    computed similarity on disc to avoid using large memory.

    The similarity is the distance L_2^2. D(i, j) = ||x_ i - x_j||_2^2.
    if the histogram proximity is used:
    D(i, j) = ||x_ i - x_j||_2^2 + proximity_histo(i, j).
    Both terms are normalized separately.
    is proximity_histo(., .) is not symmetric, D(i, j) is no longer
    symmetric. because of this asymmetry, proximity_histo(i, j) will compute
    the proximity between i, and j by considering j as the source (labeled)
    and i as target (unlabeled). this applies systematically to D(i, j).

    Applicable for both tasks: CL, SEG.
    """
    def __init__(self, task):
        """
        Init. function.

        :param task: str. CL, or SEG. classification or segmentation.
        """
        super(PairwiseSimilarity, self).__init__()

        msg = "task {} is unknown. check {}.".format(task, constants.tasks)
        assert task in constants.tasks, msg

        self.task = task

        self.sim = _ComputeSim()
        self.hist_prox = LossHistogramsMatching()

    def __call__(self, data,args, device, outd, label=None):
        """
        Compute the pairwise distance.

        :param data: list of str-path to samples. The representation of each
        sample will be computed using a projector. its config. is in args.
        :param args: object containing the the main file input arguments.
        :param device: device on where the computation will take place.
        :param outd path where we write the similarities.
        :param label: str or None. used only in the case of self.task is SEG.
        """
        already_done = False
        if already_done:  # if we already computed this sim. nothing to do.
            return 0

        histc = None
        epsilon = 1e-8
        if args.use_dist_global_hist:
            histc = SoftHistogram(bins=args.nbr_bins_histc,
                                  min=args.min_histc,
                                  max=args.max_histc,
                                  sigma=args.sigma_histc).to(device)
        # dataloader
        transform_tensor = get_transforms_tensor(args)
        set_default_seed()
        dataset = PhotoDataset(
            data,
            args.dataset,
            args.name_classes,
            transforms.Compose([transforms.ToTensor()]),
            set_for_eval=False, transform_img=None,
            resize=None,
            crop_size=None,
            padding_size=(None, None),
            padding_mode=args.padding_mode,
            up_scale_small_dim_to=None,
            do_not_save_samples=True,
            ratio_scale_patch=1.,
            for_eval_flag=True,
            scale_algo=args.scale_algo,
            resize_h_to=args.resize_h_to_opt_mask,
            resize_mask=args.resize_mask_opt_mask,  # not important.
            enhance_color=args.enhance_color,
            enhance_color_fact=args.enhance_color_fact
        )
        set_default_seed()
        data_loader = DataLoader(dataset,
                                 batch_size=args.pair_w_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 collate_fn=default_collate,
                                 worker_init_fn=_init_fn
                                 )
        set_default_seed()

        gaussian_smoother = None

        # loop! on GPU.
        nbr_samples = len(dataset)
        nbr_batches = len(data_loader)
        acc_label_prop = 0.
        z = 0.
        # project  all data and store them on disc in batches.
        idss = []
        labelss = []
        list_projections = []
        tag = ""
        # for the task SEG, the tag is helpful to avoid mixing files.
        if self.task == constants.SEG:
            tag = "_{}_{}".format(self.task, label)

        for j, (ids, imgs, masks, labels, tags, _) in enumerate(data_loader):
            with torch.no_grad():
                imgs = imgs.to(device)

                # 2. compute the histograms for matching.
                if args.use_dist_global_hist:
                    nbrs, c, h, w = imgs.shape

                    if args.smooth_img:
                        if gaussian_smoother is None:
                            gaussian_smoother = GaussianSmoothing(
                                channels=c,
                                kernel_size=args.smooth_img_ksz,
                                sigma=args.smooth_img_sigma, dim=2,
                                exact_conv=True,
                                padding_mode='reflect').to(device)
                        # smooth the image.
                        imgs = gaussian_smoother(imgs)

                    re_imgs = imgs.view(nbrs * c, h * w)
                    hists_j = histc(re_imgs)  # nbrs * c, nbr_bins
                    # normalize to prob. dist
                    hists_j = hists_j + epsilon
                    hists_j = hists_j / hists_j.sum(dim=-1).unsqueeze(1)
                    hists_j = hists_j.view(nbrs, c, -1).cpu()

                    with open(
                            join(outd, "histj_{}{}.pkl".format(
                                j, tag)), "wb") as fhist:
                        pkl.dump(hists_j, fhist, protocol=pkl.HIGHEST_PROTOCOL)

                # store some stuff.
                idss.extend(ids)
                labelss.extend(labels.numpy().tolist())
                # can't fit in memory
                # list_projections.append(copy.deepcopy(pr_j))

        # list_projections = [pr.to(device) for pr in list_projections]
        # compute sim.
        for i in tqdm.tqdm(range(nbr_samples), ncols=80, total=nbr_samples):
            id_sr, img_sr, mask_sr, label_sr, tag_sr, _ = dataset[i]
            if img_sr.ndim == 2:
                img_sr = img_sr.view(1, 1, img_sr.size()[0], img_sr.size()[1])
            elif img_sr.ndim == 3:
                img_sr = img_sr.view(1, img_sr.size()[0], img_sr.size()[1],
                                     img_sr.size()[2])
            else:
                raise ValueError('Unexpected dim: {}.'.format(img_sr.ndim))

            img_sr = img_sr.to(device)
            # histo
            histo_trg = None
            if args.use_dist_global_hist:
                if args.smooth_img:
                    img_sr = gaussian_smoother(img_sr)

                nbrs, c, h, w = img_sr.shape  # only one image.-> nbrs=1
                histo_trg = histc(img_sr.view(nbrs * c, h * w))  # c,
                # nbrbins.
                # normalize to prob. dist
                histo_trg = histo_trg + epsilon
                histo_trg = histo_trg / histo_trg.sum(dim=-1).unsqueeze(1)

            dists = None
            histo_prox = None
            for j in range(nbr_batches):
                with torch.no_grad():

                    # 2. histo proximity =======================================
                    if args.use_dist_global_hist:
                        with open(join(outd, "histj_{}{}.pkl".format(j, tag)),
                                  "rb") as fhisto:
                            hists_j = pkl.load(fhisto).to(device)  # bsize,
                            # c, nbrbins.

                        bs_sr, c_sr, nbr_bn_sr = hists_j.shape
                        tmp = self.hist_prox(
                            trg_his=histo_trg.repeat(bs_sr, 1),
                            src_his=hists_j.view(bs_sr * c_sr, -1))  # =>
                        # bs_sr * sr_c.
                        tmp = tmp.view(bs_sr, c_sr)

                        # tmp = self.sim(x, pr_j.to(device))
                        if tmp.ndim == 0:  # case of grey images with batch
                            # size of 1.
                            tmp = tmp.view(1, 1)

                        if histo_prox is None:
                            histo_prox = tmp
                        else:
                            histo_prox = torch.cat((histo_prox, tmp), dim=0)

            proximity = None
            if dists is not None:
                dists = dists.squeeze()  # remove the 1 dim. it happens when
                # batch_size == 1.
                dists = dists.cpu()
                proximity = dists.view(-1, 1)

            if histo_prox is not None:
                histo_prox = histo_prox.cpu()
            # shapes: dists: n. histo_prox: n, c where c is the number of
            # plans in the images.

            if args.use_dist_global_hist:
                # proximity = [l2 dist, r, g, b] or [l2 dist, grey]
                if proximity is not None:
                    proximity = torch.cat((proximity, histo_prox), dim=1)
                else:
                    proximity = histo_prox

            z += proximity.sum(dim=0)
            # store sims.
            srt, idx = torch.sort(
                proximity.sum(dim=1).squeeze(), descending=False)

            msg = "ERROR: {}".format(proximity[idx[0]].sum())
            # floating point issue: 1.1920928955078125e-07.
            # assert proximity[idx[0]].sum() == 0., msg

            label_pred = labelss[idx[1]]  # take the second because the first
            # is 0.
            # it is ok to overload the disc to avoid runtime cost.
            stats = {'id_sr': id_sr,  # id source
                     'label_sr': label_sr,  # label source
                     'label_pred': label_pred,
                     'nearest_id': idss[idx[1]],  # closest sample.
                     'proximity': proximity,
                     'index_sort': idx  # so we do not have to sort again. [ok]
                     }
            # name of the file: id_idNearest. this allows to get the id of
            # the nearest sample without reading the file. this speeds up the
            # pairing by avoiding disc access.
            id_nearest = stats['nearest_id']

            torch.save(proximity, join(outd, '{}.pt'.format(id_sr)))
            acc_label_prop += (label_sr == label_pred) * 1.

            if args.task == constants.SEG:
                msg = 'for weakly.sup.seg, all samples of the data provided' \
                      'to this function must have the same label. it does ' \
                      'not seem the case.W'
                assert label_sr == label_pred, msg

        # Cleaning.
        for j in range(nbr_batches):
            path1 = join(outd, "histj_{}{}.pkl".format(j, tag))
            path2 = join(outd, "histj_{}{}.pkl".format(j, tag))
            for path in [path1, path2]:
                if os.path.isfile(path):
                    os.remove(path)

        # store accuracy: the upper bound perf (when every sample is labeled
        # except one). this is useful only for classification task only.
        shared_stats = {'idss': idss,
                        'labelss': labelss,
                        'acc': 100. * acc_label_prop / nbr_samples,
                        'z': z.cpu()}
        with open(join(
                outd, 'shared-stats{}.pkl'.format(tag)), 'wb') as fout:
            pkl.dump(shared_stats, fout,  protocol=pkl.HIGHEST_PROTOCOL)

        if args.task == constants.SEG:
            msg = 'for weakly.sup.seg, accuracy is expected to be 100%. but' \
                  'found {}'.format(shared_stats['acc'])
            assert shared_stats['acc'] == 100., msg

        announce_msg(
            'Upper bound classification accuracy: {}%'.format(
                shared_stats['acc']
            ))
        announce_msg('Z: {}'.format(z))
