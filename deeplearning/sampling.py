import sys
import random
import os
from os.path import join
from copy import deepcopy
import glob
from os.path import basename
import datetime as dt

import pickle as pkl
import numpy as np
import tqdm
import torch

sys.path.append("..")
from reproducibility import set_default_seed
import constants
from tools import chunk_it

__all__ = ["Sampler", "PairSamples"]


class Sampler(object):
    """
    Class that performs sampling for active learning.
    """
    def __init__(self,
                 al_type,
                 p_samples,
                 p_init_samples,
                 device,
                 task
                 ):
        """
        Init, function.
        :param al_type: active learning method. it determines the selection
        criterion.
        :param p_samples: int, number of samples to select at each round.
        when the class label is known (SEG task), this number if divided by the
        number of classes to balance the sampling across the classes. (case of
        weakly supervised segmentation.). this number is fix across all the
        active learning rounds because it is computed over the number of
        samples of the entire original trainset [full dataset].
        :param p_init_samples: float ]0, 100]. percentage of samples to
            select from each class to initiate active learning (round 0).
            This number is divided by the number of classes to void any bias
            that comes from unbalanced classes.
        :param device: device where the computation will be run for the case of
        'ours'. GPU is preferable to do some matrix multiplications.
        :param task: str. type of the task: CL, SEG.
        """
        super(Sampler, self).__init__()

        msg = "Unrecognized task {} from {}".format(task, constants.tasks)
        assert task in constants.tasks, msg
        self.task = task
        msg = "'p_samples' must be > 0. found {}".format(p_samples)
        assert p_samples > 0., msg

        self.p_samples = p_samples

        msg = "'p_init_samples' must be > 0. found {}".format(p_init_samples)
        assert p_init_samples > 0., msg
        msg = "'p_init_samples' must be <= 100. found {}".format(p_init_samples)
        assert p_init_samples <= 100., msg

        self.p_init_samples = p_init_samples
        self.device = device

        msg = "'al_type' must be in [{}]. found {}".format(constants.al_types,
                                                           al_type)
        assert al_type in constants.al_types, msg

        self.al_type = al_type

    def sample_init_random_samples(self, tr_samples):
        """
        Get the initial random training samples for active learning.
        This sampling function has to be deterministic with respect to the
        initial seed. All active learning methods will call this function as
        round 0. sampling is done to select balanced classes. while this is not
        realistic, it is useful to avoid the bias of unbalanced classes at the
        first round.

        This rule is applied for both tasks: CL, SEG.

        :param tr_samples: list of str. contains the samples to sample from.
            each sample has the following form:
            "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4
            Possible tags:
            0: labeled
            1: unlabeled
            2: labeled but came from unlabeled set.
            see create_folds.py
        """
        # Applicable for both tasks: CL, SEG.

        set_default_seed()
        msg = "'tr_samples' must be a list. found {}".format(type(tr_samples))
        assert isinstance(tr_samples, list), msg

        labels = [x[3] for x in tr_samples]
        sz = len(labels)
        unique_l = sorted(list(set(labels)))  # set is not deterministic...
        nbr_cls = len(unique_l)
        nbr = int(self.p_init_samples * sz / 100. / nbr_cls)
        msg = "{} samples will be sampled from each class. fix" \
              " p_init: {}.".format(nbr, self.p_init_samples)
        assert nbr > 0, msg
        samples = []

        for lb in unique_l:
            idx = [i for i, x in enumerate(labels) if x == lb]
            assert len(idx) > 0, 'Something is wrong. We expect non-zero list.'
            for j in range(100):
                random.shuffle(idx)
            samples.extend([deepcopy(tr_samples[z]) for z in idx[:nbr]])

        for i in range(100):
            random.shuffle(samples)

        set_default_seed()
        # change the tag to labeled:
        for i, _ in enumerate(samples):
            samples[i][4] = constants.L

        return samples

    def random_selection(self, tr_samples):
        """
        Select samples randomly.
        For the task SEG: since the global labels are know, we samples
        balanced classes (we sample the same number of samples from each class.)

        :param tr_samples: list of samples. used to select samples
        to be labeled. (i.e. this is the list of unlabeled samples)  see
        dataloader. This set is the leftovers. also, noted U.
        """
        if len(tr_samples) == 0:
            return []

        set_default_seed()

        if self.task == constants.CL:
            idx = list(range(len(tr_samples)))

            nbr = min(self.p_samples, len(tr_samples))
            assert nbr > 0, "nbr=0."

            for i in range(1000):  # remove any bias in the order.
                random.shuffle(idx)

            samples = [deepcopy(tr_samples[i]) for i in idx[:nbr]]

        elif self.task == constants.SEG:

            labels = [x[3] for x in tr_samples]
            sz = len(labels)
            unique_l = sorted(list(set(labels)))  # set is not deterministic...
            nbr_cls = len(unique_l)
            nbr = int(min(self.p_samples, sz) / nbr_cls)
            # if the number of leftover is less than p_samples, we take one
            # IF possible (dealt with later).
            if nbr == 0:
                nbr = 1

            msg = "{} samples will be sampled from each class. fix" \
                  " p_init: {}.".format(nbr, self.p_init_samples)
            assert nbr > 0, msg
            samples = []

            for lb in unique_l:
                idx = [i for i, x in enumerate(labels) if x == lb]
                if len(idx) > 0:  # sample from this class only if there are
                    # samples.
                    for j in range(100):
                        random.shuffle(idx)
                    samples.extend([deepcopy(tr_samples[z]) for z in idx[:nbr]])
        else:
            raise ValueError("Unknown task {}. see {}.".format(
                self.task, constants.tasks))

        # change the tag to labeled:
        set_default_seed()
        for i, _ in enumerate(samples):
            samples[i][4] = constants.L

        return samples

    def entropy_selection(self, tr_samples, exp_fd):
        """
        Selection samples that have high entropy.
        :param tr_samples: list of samples. used to select samples
        to be labeled. (i.e. this is the list of unlabeled samples)  see
        dataloader. This set is the leftovers. also, noted U.
        """
        if len(tr_samples) == 0:
            return []

        set_default_seed()
        nbr = min(self.p_samples, len(tr_samples))
        assert nbr > 0, "nbr=0."

        path_f = join(exp_fd, 'leftovers', 'entropy-trainset-leftovers.pkl')
        assert os.path.exists(path_f), "{} does not exist.".format(path_f)
        with open(path_f, 'rb') as fin:
            stats = pkl.load(fin)

        entropy_all = stats['entropy']
        ids_all = stats['ids']

        if self.task == constants.CL:
            u_ids = [s[0] for s in tr_samples]
            u_entropy = []

            for id_ in u_ids:
                u_entropy.append(entropy_all[ids_all.index(id_)])

            u_entropy = np.array(u_entropy)
            idx = np.argsort(u_entropy)[::-1][:nbr]

            for i in range(1000):  # remove any bias in the order.
                random.shuffle(idx)

            samples = [deepcopy(tr_samples[i]) for i in idx]

        elif self.task == constants.SEG:
            labels = [x[3] for x in tr_samples]
            sz = len(labels)
            unique_l = sorted(list(set(labels)))  # set is not deterministic...
            nbr_cls = len(unique_l)
            nbr = int(min(self.p_samples, sz) / nbr_cls)
            # if the number of leftover is less than p_samples, we take one
            # IF possible (dealt with later).
            if nbr == 0:
                nbr = 1

            msg = "{} samples will be sampled from each class. fix" \
                  " p_init: {}.".format(nbr, self.p_init_samples)
            assert nbr > 0, msg
            samples = []

            for lb in unique_l:
                idx = [
                    tr_samples[i][0] for i, x in enumerate(labels) if x == lb
                ]

                if len(idx) > 0:  # sample from this class only if there are
                    # samples.

                    u_entropy = []

                    for id_ in idx:
                        u_entropy.append(entropy_all[ids_all.index(id_)])

                    u_entropy = np.array(u_entropy)
                    idx2 = np.argsort(u_entropy)[::-1][:nbr]

                    idx_selected = [idx[zz] for zz in idx2]

                    samples.extend([deepcopy(z) for z in tr_samples if z[0]
                                    in idx_selected])
        else:
            raise ValueError("Unknown task {}. see {}.".format(
                self.task, constants.tasks))

        # change the tag to labeled:
        set_default_seed()
        for i, _ in enumerate(samples):
            samples[i][4] = constants.L

        print('entropy selection')

        for s in samples:
            print(s)

        print('entropy selection')

        return samples

    def mc_dropout_selection(self, tr_samples, exp_fd):
        """
        Selection samples that have high variance based on mc-dropout.
        :param tr_samples: list of samples. used to select samples
        to be labeled. (i.e. this is the list of unlabeled samples)  see
        dataloader. This set is the leftovers. also, noted U.
        """
        if len(tr_samples) == 0:
            return []

        set_default_seed()
        nbr = min(self.p_samples, len(tr_samples))
        assert nbr > 0, "nbr=0."

        path_f = join(exp_fd, 'leftovers', 'mc-dropout-trainset-leftovers.pkl')
        assert os.path.exists(path_f), "{} does not exist.".format(path_f)
        with open(path_f, 'rb') as fin:
            stats = pkl.load(fin)

        variance_all = stats['mc-dropout-var']
        ids_all = stats['ids']

        if self.task == constants.CL:
            u_ids = [s[0] for s in tr_samples]
            u_mc_dropout = []

            for id_ in u_ids:
                u_mc_dropout.append(variance_all[ids_all.index(id_)])

            u_mc_dropout = np.array(u_mc_dropout)
            idx = np.argsort(u_mc_dropout)[::-1][:nbr]

            for i in range(1000):  # remove any bias in the order.
                random.shuffle(idx)

            samples = [deepcopy(tr_samples[i]) for i in idx]

        elif self.task == constants.SEG:

            labels = [x[3] for x in tr_samples]
            sz = len(labels)
            unique_l = sorted(list(set(labels)))  # set is not deterministic...
            nbr_cls = len(unique_l)
            nbr = int(min(self.p_samples, sz) / nbr_cls)
            # if the number of leftover is less than p_samples, we take one
            # IF possible (dealt with later).
            if nbr == 0:
                nbr = 1

            msg = "{} samples will be sampled from each class. fix" \
                  " p_init: {}.".format(nbr, self.p_init_samples)
            assert nbr > 0, msg
            samples = []

            for lb in unique_l:
                idx = [
                    tr_samples[i][0] for i, x in enumerate(labels) if x == lb
                ]
                if len(idx) > 0:  # sample from this class only if there are
                    # samples.
                    u_mc_dropout = []

                    for id_ in idx:
                        u_mc_dropout.append(variance_all[ids_all.index(id_)])

                    u_mc_dropout = np.array(u_mc_dropout)
                    idx2 = np.argsort(u_mc_dropout)[::-1][:nbr]
                    idx_selected = [idx[zz] for zz in idx2]

                    samples.extend([deepcopy(z) for z in tr_samples if z[0]
                                    in idx_selected])
        else:
            raise ValueError("Unknown task {}. see {}.".format(
                self.task, constants.tasks))

        # change the tag to labeled:
        set_default_seed()
        for i, _ in enumerate(samples):
            samples[i][4] = constants.L

        return samples

    def our_selection(self, tr_samples, unlabeled, labeled, clustering,
                      simsfd, args):
        """
        EFFICIENT IMPLEMENTATION: COUPLE OF MB OF RAM + VERY FAST (20s).
        measured over MNIST (49950 samples: takes 40GB, and 4hours with naive
        implementation). yes this implementation uses only couple of
        megabytes of memory.

        USE ITERATIVE ALGORITHM:
        DENSITY_U_i(T+1) := DENSITY(T) - D_i_selected (~0TIME, ~0RAM)
        DIVERSITY_L_i(T+1) := DIVERSITY_L_i(T) + D_i_selected (~0TIME, ~0RAM)
        DENSTY_LABELS: STORE ONLY THE KNN. (~TIME, ~RAM)

        Select samples ITERATIVELY, based on the clustering criterion:
        select sample x that has lower score, such as the score is:
          density(x, U) + diversity(x, L)
          or
          density(x, U) + density_labelling(x, U)
          or
          density(x, U) + diversity(x, L) + density_labelling(x, U)

        :param tr_samples: list of samples to select from. samples contain
        all the necessary information.
        :param unlabeled: lits of str ids of unlabeled samples to select from.
        :param labeled: list of str id of labeled samples.
        :param clustering: str, type of clustering. see
        constants.ours_clustering.
        :param simsfd: folder where the similarity measures have been stored.
        :param args: args of main.py.
        """
        raise NotImplementedError('NOT UP TO DATE.')

        # TODO: upgrade to weakly sup. segm.
        nbr = min(self.p_samples, len(tr_samples))
        assert nbr > 0, "nbr=0."

        with open(join(simsfd, 'shared-stats.pkl'), 'rb') as fin:
            shared_stats = pkl.load(fin)
        z = shared_stats['z']
        assert z != 0., 'z is zero which is not expected.'
        all_idss = shared_stats['idss']
        ordered_idss = torch.tensor([int(xx) for xx in all_idss],
                                    dtype=torch.int)
        n = len(all_idss)
        state_l = torch.zeros(n, dtype=torch.float32)  # true (1.): labeled,
        # false (0.): unlabeled.
        indx_of_id = dict()

        for i, id in enumerate(all_idss):
            indx_of_id[id] = i
            if id in labeled:
                state_l[i] = 1.
        state_u = 1. - state_l

        nbr_u = len(unlabeled)
        nbr_l = n - nbr_u
        k_d_lab = args.k_dense_labels
        diversity_ok, dens_labls_ok = False, False
        if clustering in [
            constants.CLUSTER_DENSITY_DIVERSITY,
                constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:
            diversity_ok = True

        if clustering in [
            constants.CLUSTER_DENSITY_LABELLING,
                constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:
            assert args.k_dense_labels != 0, 'args.k_dense_labels is ' \
                                             'zero. not allowed.'
            dens_labls_ok = True

        scores = torch.zeros(nbr_u, dtype=torch.float32)
        density = torch.zeros(nbr_u, dtype=torch.float32)

        if diversity_ok:
            diversity = torch.zeros(nbr_u, dtype=torch.float32)
        if dens_labls_ok:
            sorted_neigh = torch.zeros((nbr_u, k_d_lab), dtype=torch.int)
            sorted_neigh_bin_l = torch.zeros(
                (nbr_u, k_d_lab), dtype=torch.float32)

        u_id_tracker = []
        position = torch.zeros(nbr_u, dtype=torch.int64)
        nbr_u = float(nbr_u)
        nbr_l = float(nbr_l)
        k_d_lab = float(k_d_lab)

        for i, id in tqdm.tqdm(enumerate(unlabeled), ncols=80, total=len(
                unlabeled)):
            u_id_tracker.append(id)
            position[i] = indx_of_id[id]
            dist = torch.load(join(simsfd, '{}.pt'.format(id)))
            _, srt_idx = torch.sort(dist, descending=False)

            dist = dist / z
            # density
            density[i] = (dist * state_u).sum() / (nbr_u - 1.)
            scores[i] += density[i]
            # diversity
            if diversity_ok:
                diversity[i] = ((1. - dist) * state_l).sum() / nbr_l
                scores[i] += diversity[i]

            # labels density
            # skip index 0 because it is the sample i itself.
            if dens_labls_ok:
                sorted_neigh[i] = ordered_idss[
                    srt_idx[1:args.k_dense_labels + 1]]
                sorted_neigh_bin_l[i] = state_l[
                    srt_idx[1:args.k_dense_labels + 1]].bool()
                scores[i] += sorted_neigh_bin_l[i].sum() / k_d_lab

        # starts iterative selection.
        selected = []  # contains the ids to select.
        for _ in tqdm.tqdm(range(nbr), ncols=80, total=nbr):
            nbr_left = len(u_id_tracker)

            if nbr_left == 1:  # add this sample and get out.
                min_id = u_id_tracker[0]
                assert min_id not in selected, 'something is wrong'
                selected.append(min_id)
                break

            if nbr_left == 0:  # get out.
                break

            # now the score of each sample has been compute, we can find the
            # best sample to select by taking the sample with the lowest score.
            _, idxbest = torch.sort(scores, descending=False)
            min_id = u_id_tracker[idxbest[0]]
            selected.append(min_id)

            # updates after selection
            state_l[indx_of_id[min_id]] = 1.  # not used
            state_u[indx_of_id[min_id]] = 0.  # not used

            selc = [gg for gg in range(int(nbr_u)) if gg != idxbest[0]]  # drop
            # the selected sample.

            # since the similarity is symmetric, we do not need to load every
            # sample to get their distance to the selected sample.
            dist_selected = torch.load(join(simsfd, '{}.pt'.format(min_id)))

            position = position[selc[:]]
            density = density[selc[:]]
            # needs to remove the normalization
            density = density * (nbr_u - 1.) - dist_selected[position[:]]
            if nbr_u > 2:  # normalize if we have more than 2 samples.
                # otherwise, we do have exactly two samples. and the
                # norlalization factor is 1.
                density = density / (nbr_u - 2.)  # renormalize

            # recompute scores from scratch
            scores = density

            if diversity_ok:
                diversity = diversity[selc[:]]
                diversity = diversity * nbr_l + dist_selected[position[:]]
                diversity = diversity / (nbr_l + 1)  # renormalize
                scores += diversity

            if dens_labls_ok:
                sorted_neigh_bin_l = sorted_neigh_bin_l[selc[:], :]
                sorted_neigh = sorted_neigh[selc[:], :]

                # update
                sorted_neigh_bin_l += (sorted_neigh == min_id)

                scores += sorted_neigh_bin_l.sum(dim=1) / k_d_lab

            nbr_u -= 1.
            nbr_l += 1.
            del u_id_tracker[idxbest[0]]  # remove min_id

        for i in range(1000):  # remove any bias in the order.
            random.shuffle(selected)

        # return the selected samples
        samples = []
        for sam in tr_samples:
            if sam[0] in selected:
                samples.append(deepcopy(sam))
                samples[-1][4] = constants.L  # change the tag to labeled:

        return samples

    def our_selection_slow(self, tr_samples, unlanbeled, labeled,
                           clustering, simsfd, args):
        """
        warning: this is extremely slow...

        Select samples ITERATIVELY, based on the clustering criterion:
        select sample x that has lower score, such as the score is:
          density(x, U) + diversity(x, L)
          or
          density(x, U) + density_labelling(x, U)
          or
          density(x, U) + diversity(x, L) + density_labelling(x, U)

        :param tr_samples: list of samples to select from. samples contain
        all the necessary information.
        :param unlanbeled: lits of str ids of unlabeled samples to select from.
        :param labeled: list of str id of labeled samples.
        :param clustering: str, type of clustering. see
        constants.ours_clustering.
        :param simsfd: folder where the similarity measures have been stored.
        :param args: args of main.py.
        """
        raise NotImplementedError('NOT UP TO DATE.')

        # TODO: upgrade to weakly sup. segm.
        nbr = min(self.p_samples, len(tr_samples))
        assert nbr > 0, "nbr=0."

        with open(join(simsfd, 'shared-stats.pkl'), 'rb') as fin:
            shared_stats = pkl.load(fin)
        z = shared_stats['z']
        assert z != 0., 'z is zero which is not expected.'
        all_idss = shared_stats['idss']
        n = len(all_idss)
        state = np.zeros(n, dtype=float)  # true: labeled, false: unlabeled.
        indx_of_id = dict()
        for i, id in enumerate(all_idss):
            indx_of_id[id] = i
            if id in labeled:
                state[i] = 1.

        dist_u = dict()  # distance of unlabeled samples only.
        idx_sort = dict()
        # load all the distance in U.
        # overloading the memory... this is not really a good idea. but since
        # we are doing this function at the end of main.py, there is no
        # risque (multi-process-wise). but seriously, it is a terrible idea.

        # we are not going to overload the memory, but load when needed. but
        # we will do tens of disc reading.
        unlab_dict_ids = dict((key, None) for key in unlanbeled)

        for id in tqdm.tqdm(unlanbeled, ncols=80, total=len(unlanbeled)):

            dist = torch.load(join(simsfd, '{}.pt'.format(id)))
            srt, srt_idx = torch.sort(dist, descending=False)
            dist_u[id] = deepcopy(dist) / z
            idx_sort[id] = deepcopy(srt_idx)

        # stats iterative selection.
        selected = []  # contains the ids to select.
        for _ in tqdm.tqdm(range(nbr), ncols=80, total=nbr):
            scores_iter = dict()
            nbr_l = state.sum()
            nbr_u = (1. - state).sum()

            nbr_left = len(list(dist_u.keys()))

            if nbr_left == 1:  # add this sample and get out.
                min_id = list(dist_u.keys())[0]
                assert min_id not in selected, 'something is wrong'
                selected.append(min_id)
                break

            if nbr_left == 0:  # get out.
                break

            for k in dist_u.keys():  # allows to select only one sample.
                s = 0.
                # this file will be loaded: nbr times. in total, we reload
                # : nbr * nbr_unlabeled files.
                # dist = torch.load(join(simsfd, '{}.pt'.format(k)))
                dist = dist_u[k]
                srt_idx = idx_sort[k]
                # _, srt_idx = torch.sort(dist, descending=False)
                # density
                s += (dist * (state - 1)).sum() / (nbr_u - 1)  # E[dens_u]
                # diversity
                if clustering in [
                    constants.CLUSTER_DENSITY_DIVERSITY,
                        constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:
                    s += ((1. - dist) * state).sum() / nbr_l  # E[div_l]

                # labels density
                if clustering in [
                    constants.CLUSTER_DENSITY_LABELLING,
                        constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:

                    assert args.k_dense_labels != 0, 'args.k_dense_labels is ' \
                                                     'zero. not allowed.'
                    # don't take the first element of sort because it is the
                    # sample itself.
                    tmp = [state[j] for j in srt_idx[1:(
                            args.k_dense_labels + 1)]]
                    s += sum(tmp) / float(args.k_dense_labels)  # E[dens_labels]

                scores_iter[k] = s
            # select the sample with the minimum score.
            min_id = min(scores_iter, key=scores_iter.get)
            assert min_id not in selected, 'something is wrong'
            selected.append(min_id)

            # remove the selected sample from U
            del dist_u[min_id]
            # set this sample as L
            state[indx_of_id[min_id]] = 1.

        # return the selected samples
        samples = []
        for sam in tr_samples:
            if sam[0] in selected:
                samples.append(deepcopy(sam))
                samples[-1][4] = constants.L  # change the tag to labeled:

        return samples

    def our_selection_stil_slow(self, tr_samples, unlanbeled, labeled,
                                clustering, simsfd, args):
        """
        warning: this is fast but requires large RAM.

        Select samples ITERATIVELY, based on the clustering criterion:
        select sample x that has lower score, such as the score is:
          density(x, U) + diversity(x, L)
          or
          density(x, U) + density_labelling(x, U)
          or
          density(x, U) + diversity(x, L) + density_labelling(x, U)

        :param tr_samples: list of samples to select from. samples contain
        all the necessary information.
        :param unlanbeled: lits of str ids of unlabeled samples to select from.
        :param labeled: list of str id of labeled samples.
        :param clustering: str, type of clustering. see
        constants.ours_clustering.
        :param simsfd: folder where the similarity measures have been stored.
        :param args: args of main.py.
        """
        raise NotImplementedError('NOT UP TO DATE.')

        # TODO: upgrade to weakly sup. segm.
        nbr = min(self.p_samples, len(tr_samples))
        assert nbr > 0, "nbr=0."

        with open(join(simsfd, 'shared-stats.pkl'), 'rb') as fin:
            shared_stats = pkl.load(fin)
        z = shared_stats['z']
        assert z != 0., 'z is zero which is not expected.'
        all_idss = shared_stats['idss']
        n = len(all_idss)
        state = torch.zeros(n, dtype=torch.bool)  # true: labeled, false:
        # unlabeled.
        indx_of_id = dict()
        for i, id in enumerate(all_idss):
            indx_of_id[id] = i
            if id in labeled:
                state[i] = True

        nbr_u = len(unlanbeled)
        # create distance
        dist_u = torch.zeros((nbr_u, n), dtype=torch.float32)  # distance of
        # unlabeled samples only.
        sorted_neigh_bin = torch.zeros((nbr_u, n), dtype=torch.bool)  # contains
        # the state of the sorted neighbors (from closest to the furthest).
        # true if the neighbor is labeled

        sorted_neigh = torch.zeros((nbr_u, n), dtype=torch.int)  # contains
        # the indices of the sorted neighbors (from closest to the furthest).
        # true if the neighbor is labeled

        # using dist_u, sorted_neigh_bin, sorted_neigh we can compute the
        # score of each sample under a matrix form to avoid iterating over
        # all the samples.

        u_id_tracker = []
        for i, id in tqdm.tqdm(enumerate(unlanbeled), ncols=80, total=len(
                unlanbeled)):

            u_id_tracker.append(id)
            dist = torch.load(join(simsfd, '{}.pt'.format(id)))
            _, srt_idx = torch.sort(dist, descending=False)

            dist_u[i, :] = dist / z
            sorted_neigh[i, :] = srt_idx
            sorted_neigh_bin[i, :] = state[srt_idx].bool()

        # stats iterative selection.
        selected = []  # contains the ids to select.
        k_d_lab = float(args.k_dense_labels)
        for _ in tqdm.tqdm(range(nbr), ncols=80, total=nbr):
            h, w = dist_u.shape
            nbr_u = float(len(u_id_tracker))  # current number of U.
            nbr_l = float(w - nbr_u)  # current number of L.

            # ==================================================================
            #            COMPUTE THE  SCORE OF EACH SAMPLE
            #            USING ONLY MATRIX OPERATIONS
            # ==================================================================
            state_duplicated = state.repeat(h, 1)
            # density: E[density_u]
            print('density')
            scores = (dist_u * (~state_duplicated)).sum(dim=1) / (nbr_u - 1.)

            # diversity: E[div_l]
            if clustering in [
                constants.CLUSTER_DENSITY_DIVERSITY,
                    constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:
                print('diversity')
                scores += ((1. - dist_u) * state_duplicated).sum(dim=1) / nbr_l

            # label density: E[density_labels]
            if clustering in [
                constants.CLUSTER_DENSITY_LABELLING,
                    constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:
                assert args.k_dense_labels != 0, 'args.k_dense_labels is ' \
                                                 'zero. not allowed.'
                print('dense labels')
                scores += sorted_neigh_bin[
                          1:args.k_dense_labels + 1].sum(dim=1) / k_d_lab

            # now the score of each sample has been compute, we can find the
            # best sample to select by taking the sample with the lowest score.
            _, idxbest = torch.sort(scores, descending=False)
            min_id = u_id_tracker[idxbest[0]]
            selected.append(min_id)

            # remove the selected sample from the U set
            state[indx_of_id[min_id]] = True
            selc = [gg for gg in range(h) if gg != idxbest[0]]  # drop the
            # selected sample.

            dist_u = dist_u[selc[:], :]
            sorted_neigh_bin = sorted_neigh_bin[selc[:], :]
            sorted_neigh = sorted_neigh[selc[:], :]
            # update sorted neigh BIN: True + True = True, True + False = True.
            sorted_neigh_bin += (sorted_neigh == indx_of_id[min_id])

        # return the selected samples
        samples = []
        for sam in tr_samples:
            if sam[0] in selected:
                samples.append(deepcopy(sam))
                samples[-1][4] = constants.L  # change the tag to labeled:

        return samples

    def our_selection_gpu(self, tr_samples, unlanbeled, labeled,
                          clustering, simsfd, args):
        """
        warning: this is fast but requires large RAM.

        Select samples ITERATIVELY, based on the clustering criterion:
        select sample x that has lower score, such as the score is:
          density(x, U) + diversity(x, L)
          or
          density(x, U) + density_labelling(x, U)
          or
          density(x, U) + diversity(x, L) + density_labelling(x, U)

        :param tr_samples: list of samples to select from. samples contain
        all the necessary information.
        :param unlanbeled: lits of str ids of unlabeled samples to select from.
        :param labeled: list of str id of labeled samples.
        :param clustering: str, type of clustering. see
        constants.ours_clustering.
        :param simsfd: folder where the similarity measures have been stored.
        :param args: args of main.py.
        """
        raise NotImplementedError('NOT UP TO DATE.')

        # TODO: upgrade to weakly sup. segm.
        nbr = min(self.p_samples, len(tr_samples))
        assert nbr > 0, "nbr=0."

        with open(join(simsfd, 'shared-stats.pkl'), 'rb') as fin:
            shared_stats = pkl.load(fin)
        z = shared_stats['z']
        assert z != 0., 'z is zero which is not expected.'
        all_idss = shared_stats['idss']
        n = len(all_idss)
        state = torch.zeros(n, dtype=torch.bool)  # true: labeled, false:
        # unlabeled.
        indx_of_id = dict()
        for i, id in enumerate(all_idss):
            indx_of_id[id] = i
            if id in labeled:
                state[i] = True

        nbr_u = len(unlanbeled)
        # create distance
        dist_u = torch.zeros((nbr_u, n), dtype=torch.float32)  # distance of
        # unlabeled samples only.
        sorted_neigh_bin = torch.zeros((nbr_u, n), dtype=torch.bool)  # contains
        # the state of the sorted neighbors (from closest to the furthest).
        # true if the neighbor is labeled

        sorted_neigh = torch.zeros((nbr_u, n), dtype=torch.int)  # contains
        # the indices of the sorted neighbors (from closest to the furthest).
        # true if the neighbor is labeled

        # using dist_u, sorted_neigh_bin, sorted_neigh we can compute the
        # score of each sample under a matrix form to avoid iterating over
        # all the samples.

        u_id_tracker = []
        for i, id in tqdm.tqdm(enumerate(unlanbeled), ncols=80, total=len(
                unlanbeled)):

            u_id_tracker.append(id)
            dist = torch.load(join(simsfd, '{}.pt'.format(id)))
            _, srt_idx = torch.sort(dist, descending=False)

            dist_u[i, :] = dist / z
            sorted_neigh[i, :] = srt_idx
            sorted_neigh_bin[i, :] = state[srt_idx].bool()

        # send to device (GPU preferably)
        # dist_u = dist_u.to(self.device)
        # sorted_neigh_bin = sorted_neigh_bin.to(self.device)
        sorted_neigh = sorted_neigh.to(self.device)

        bsz = 500

        # stats iterative selection.
        selected = []  # contains the ids to select.
        k_d_lab = float(args.k_dense_labels)
        for _ in tqdm.tqdm(range(nbr), ncols=80, total=nbr):
            h, w = dist_u.shape
            nbr_u = float(len(u_id_tracker))  # current number of U.
            nbr_l = float(w - nbr_u)  # current number of L.
            # ==================================================================
            #            COMPUTE THE  SCORE OF EACH SAMPLE
            #            USING ONLY MATRIX OPERATIONS
            # ==================================================================
            scores = None
            batches_it = list(range(h))
            for pt in chunk_it(batches_it, bsz):
                tmp_scores = None

                tmp_dist_u = dist_u[pt[:], :].to(self.device)
                tmp_sorted_neigh_bin = sorted_neigh_bin[
                                       pt[:], :].to(self.device)

                state_duplicated = state.repeat(len(pt), 1)
                # send to device
                state_duplicated = state_duplicated.to(self.device)
                # density: E[density_u]
                t0 = dt.datetime.now()
                tmp_scores = (tmp_dist_u * (
                    ~state_duplicated)).sum(dim=1) / (nbr_u - 1.)

                # diversity: E[div_l]
                if clustering in [
                    constants.CLUSTER_DENSITY_DIVERSITY,
                        constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:

                    tmp_scores += ((1. - tmp_dist_u) * state_duplicated).sum(
                        dim=1) / nbr_l

                # label density: E[density_labels]
                if clustering in [
                    constants.CLUSTER_DENSITY_LABELLING,
                        constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:
                    assert args.k_dense_labels != 0, 'args.k_dense_labels is ' \
                                                     'zero. not allowed.'
                    tmp_scores += tmp_sorted_neigh_bin[
                              1:args.k_dense_labels + 1].sum(dim=1) / k_d_lab

                if scores is None:
                    scores = tmp_scores
                else:
                    scores = torch.cat((scores, tmp_scores))
            # now the score of each sample has been compute, we can find the
            # best sample to select by taking the sample with the lowest score.
            _, idxbest = torch.sort(scores, descending=False)
            min_id = u_id_tracker[idxbest[0]]
            selected.append(min_id)

            # remove the selected sample from the U set
            state[indx_of_id[min_id]] = True
            selc = [gg for gg in range(h) if gg != idxbest[0]]  # drop the
            # selected sample.

            dist_u = dist_u[selc[:], :]
            sorted_neigh_bin = sorted_neigh_bin[selc[:], :]
            sorted_neigh = sorted_neigh[selc[:], :]
            # update sorted neigh BIN: True + True = True, True + False = True.
            sorted_neigh_bin += (sorted_neigh == indx_of_id[min_id])

        # return the selected samples
        samples = []
        for sam in tr_samples:
            if sam[0] in selected:
                samples.append(deepcopy(sam))
                samples[-1][4] = constants.L  # change the tag to labeled:

        return samples

    def __call__(self, tr_samples, args, ids_current_tr, simsfd, exp_fd):
        """
        Perform a selection.
        :param tr_samples: list of training samples. used to select samples
        to be labeled. (i.e. this is the list of unlabeled samples)  see
        dataloader. This set is the leftovers. also, noted U.
        :param args: object contains the passed arguments to the main. see
        main.py
        :param ids_current_tr: list of str ids of all the samples fully
        labeled used in the current active learning round. there is no sample
        in this set that belongs to `tr_samples`. noted L. U + L = entire set.
        :param simsfd: folder where the similarity measures have been stored.
        :param exp_fd: str, path to the exp folder.
        """
        msg = "Unknown selection method: {}. must be in [{}].".format(
            args.al_type, constants.al_types
        )
        assert args.al_type in constants.al_types, msg

        samples = []
        if len(tr_samples) == 1:
            samples.append(tr_samples[0])
            return samples
        elif len(tr_samples) == 0:
            return []

        if args.al_type == constants.AL_RANDOM:
            return self.random_selection(tr_samples=tr_samples)

        elif args.al_type == constants.AL_LP:
            msg = "clustering type unknow {}. " \
                  "must be in constants.ours_clusering".format(args.clustering)
            assert args.clustering in constants.ours_clustering, msg

            clustering = args.clustering
            # Random sampling
            if clustering == constants.CLUSTER_RANDOM:
                return self.random_selection(tr_samples=tr_samples)
            # Entropy-based sampling
            if clustering == constants.CLUSTER_ENTROPY:
                return self.entropy_selection(
                    tr_samples=tr_samples, exp_fd=exp_fd)

            if clustering in [constants.CLUSTER_DENSITY_DIVERSITY,
                              constants.CLUSTER_DENSITY_LABELLING,
                              constants.CLUSTER_DENSITY_DIVERSITY_LABELLING]:
                ids_unlabeled = [z[0] for z in tr_samples]
                return self.our_selection(
                    tr_samples=tr_samples,
                    unlabeled=ids_unlabeled,
                    labeled=ids_current_tr,
                    clustering=args.clustering,
                    simsfd=simsfd,
                    args=args
                    )
            else:
                raise ValueError(
                    'Ours, but unknown clustering {}'.format(clustering))

        elif args.al_type == constants.AL_ENTROPY:
            return self.entropy_selection(tr_samples=tr_samples, exp_fd=exp_fd)

        elif args.al_type == constants.AL_MCDROPOUT:
            return self.mc_dropout_selection(
                tr_samples=tr_samples, exp_fd=exp_fd)
        else:
            raise ValueError(
                "Unknown active learning selection: {}".format(args.al_type))


class PairSamples(object):
    """
    Class that pairs samples based on similarity.
    """
    def __init__(self, task, knn=1):
        """
        Init. function.
        :param task: str. type of the task. see constants.py
        :param knn: int. the max number of neighbors allowed to be explored.
        """
        super(PairSamples, self).__init__()

        msg = "unknown 'task'= {}. supported tasks={}".format(
            task, constants.tasks)
        assert task in constants.tasks, msg
        self.task = task

        msg = "'knn' must be int, found {}".format(type(knn))
        assert isinstance(knn, int), msg
        msg = "'knn' must be > 1, found {}".format(knn)
        assert knn > 0, msg

        self.knn = knn

    def __call__(self, labeled, unlabeled, simsfd):
        """
        Pair each sample from the unlabeled set to the closest sample in the
        labeled set.

        :param labeled: list of labeled samples.
        :param unlabeled: list of labeled samples.
        :param simsfd: folder where the similarity measures have been stored.

        :return pairs_labeled: dict where each key is the id of a sample in
                unlabeled set,
                and its value is the id of the sample from the labeled set that
                is paired with <dangling preposition>. (unlabeled, labeled)
        """
        msg = "'labeled' must be a list. found {}".format(type(labeled))
        assert isinstance(labeled, list), msg
        msg = "'unlabeled' must be a list. found {}".format(type(unlabeled))
        assert isinstance(unlabeled, list), msg

        ids_lab = [z[0] for z in labeled]  # ids of labeled samples.

        all_samples = labeled + unlabeled
        id_all_samples = [sx[0] for sx in all_samples]

        pairs = dict()
        nbr = len(unlabeled)
        if self.task == constants.CL:
            with open(join(simsfd, 'shared-stats.pkl'), 'rb') as fin:
                shared_stats = pkl.load(fin)

            all_idss = shared_stats['idss']
            z = shared_stats['z'].item()  # constant scalar.

        for s in tqdm.tqdm(unlabeled, ncols=80, total=nbr):
            id = s[0]
            if self.task == constants.SEG:
                label = s[3]
                tag = "_{}_{}".format(self.task, label)
                with open(join(
                        simsfd,
                        'shared-stats{}.pkl'.format(tag)), 'rb') as fin:
                    shared_stats = pkl.load(fin)

                all_idss = shared_stats['idss']
                z = shared_stats['z'].view(1, -1)
            # expensive: disc access
            proximity = torch.load(join(simsfd, '{}.pt'.format(id)))
            proximity = proximity / z
            proximity = proximity.sum(dim=1).view(-1)
            srt, idx = torch.sort(proximity, descending=False)

            # in this case as well, we take all the samples.
            nearest_ids = [all_idss[j] for j in idx[1:]]

            # pairing to labeled data only: search for the closest labeled
            # sample. this is a simple version of label propagation.

            nbr_knn = 0
            for id_n in nearest_ids:
                if self.task == constants.CL:
                    if id_n in ids_lab:
                        pairs[id] = id_n
                        break

                    nbr_knn += 1
                    if nbr_knn == self.knn:
                        break

                elif self.task == constants.SEG:
                    # find the global label of id_n: we have access to the
                    # global of every sample.
                    label_l = all_samples[id_all_samples.index(id_n)][3]
                    # we pair samples from the same class in SEG task.
                    if label == label_l:
                        if id_n in ids_lab:
                            pairs[id] = id_n
                            break

                        nbr_knn += 1
                        if nbr_knn == self.knn:  # if we exhaust the search
                            # limit, we leave.
                            break
        return pairs

