import os
from os.path import join
from copy import deepcopy

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


import constants


__all__ = ["PlotActiveLearningRounds"]


class PlotActiveLearningRounds(object):
    """
    Plot stats. form all active learning rounds.
    Folder structure:
    folder
         |---common
         |---vision
         |---shared_opt_masks
         |---'exp_folder_round_0'
         |---'exp_folder_round_1'
         |---'exp_folder_round_2'
         |---'exp_folder_round_3'
         |---'exp_folder_round_4'
         |---'exp_folder_round_5'
         |---'exp_folder_round_...'
    """
    def __init__(self,
                 folder_rounds,
                 task,
                 max_al_its
                 ):
        """
        Init. function
        :param folder_rounds: str, path to the folder where all the rounds
        folders are located.
        :param task: str. type of the task (CL, SEG).
        :param max_al_its: int. maximum al rounds.
        """
        super(PlotActiveLearningRounds, self).__init__()

        msg = "folder {} does not exist".format(folder_rounds)
        assert os.path.exists(folder_rounds), msg

        msg = "'task' must be in {}. found {}.".format(constants.tasks, task)
        assert task in constants.tasks, msg

        self.task = task

        self.folder_rounds = folder_rounds
        self.current_folder_rounds = folder_rounds

        self.max_rep = max_al_its  # how many time we repeat the values in
        # the case of sull-sup since there is only one value.
        self.frequency = 2  # frequency to plot xticks.
        self.shift = 5  # how much to shift ticks top and bottom when
        # plotting values.

    def plot_curve(self,
                   xticks,
                   values,
                   path,
                   title="",
                   x_str="",
                   y_str="",
                   dpi=300,
                   ylimtop=None,
                   ylimbottom=None,
                   legend_loc="lower right",
                   x=None
                   ):
        """
        Plot a curve.

        :param xticks: list of x ticks labels.
        :param values: list or numpy.ndarray of values to plot (y)
        :param path: str, path where to save the figure.
        :param title: str, the title of the plot.
        :param x_str: str, the name of the x axis.
        :param y_str: str, the name of the y axis.
        :param dpi: int, the dpi of the image.
        :param ylimtop: int or None. when int, this value is set as the upper
               limit of y axis.
        :param ylimbottom: int or None. when int, this value is set as the lower
               limit of y axis.
        :param legend_loc: str. legend location.
        """
        assert isinstance(values, list) or isinstance(values, np.ndarray), \
            "'values' must be either a list or a numpy.ndarray. You provided " \
            "`{}` .... [NOT OK]".format(type(values))

        if isinstance(values, list):
            values = np.asarray(values)

        font_sz = 6
        shift = self.shift

        fig = plt.figure()
        if x is None:
            x = list(range(values.size))

        plt.plot(x, values)
        plt.xlabel(x_str, fontsize=font_sz)
        plt.ylabel(y_str, fontsize=font_sz)
        plt.title(title, fontsize=font_sz)
        plt.xticks(x, xticks, fontsize=font_sz)
        # major grid lines with dark grey color.
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        # minor grid lines with transparent grey color.
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-',
                 alpha=0.2)
        plt.tick_params(axis='both', labelsize=6)

        if ylimtop is not None:
            if ylimbottom is not None:
                plt.ylim(ylimbottom - shift, ylimtop + shift)
            else:
                plt.ylim(min(values) - shift, ylimtop + shift)
        elif ylimbottom is not None:
            plt.ylim(ylimbottom - shift, max(values) + shift)

        # plt.legend(loc=legend_loc)  # no legend.
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
        plt.close('all')

        del fig

    def decorate_figure(self,
                        fig,
                        path,
                        title="",
                        x_str="",
                        y_str="",
                        dpi=300,
                        ylimtop=None,
                        ylimbottom=None,
                        legend_loc="lower right",
                        xticks=None,
                        x=None
                        ):
        """
        Decorate a figure. useful when the figure contains multiple curves.
        Otherwise, use self.plot_curve().
        :param xticks:
        :param path:
        :param title:
        :param x_str:
        :param y_str:
        :param dpi:
        :param ylimtop:
        :param ylimbottom:
        :param legend_loc:
        :param x:
        :return:
        """
        font_sz = 6
        shift = self.shift

        plt.xlabel(x_str, fontsize=font_sz)
        plt.ylabel(y_str, fontsize=font_sz)
        plt.title(title, fontsize=font_sz)
        if x is not None:
            plt.xticks(x, xticks, fontsize=font_sz)
        # major grid lines with dark grey color.
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        # minor grid lines with transparent grey color.
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-',
                 alpha=0.2)
        plt.tick_params(axis='both', labelsize=6)

        if (ylimtop is not None) and (ylimbottom is not None):
            plt.ylim(ylimbottom - shift, ylimtop + shift)

        plt.legend(loc=legend_loc)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
        plt.close('all')

        del fig

    def plot_metric(self,
                    lstats,
                    metric_key,
                    metric_tag,
                    ylimtop=None,
                    ylimbottom=None
                    ):
        """
        Plot a metric over: train, valid, and test.
        :param lstats: list of dicts.
        :param metric_key: str. {'acc', 'dice_idx'}.
        :param metric_tag: str. name of the metric for the title of the figure.
        :param ylimtop: int or None. when int, this value is set as the upper
               limit of y axis.
        :param ylimbottom: int or None. when int, this value is set as the lower
               limit of y axis.
        """
        nbr_sup = []
        tr, vl, tst = [], [], []

        for el in lstats:
            nbr_sup.append(el['propagation']['nbr_fully_sup'])

            tr.append(el['train'][metric_key])
            vl.append(el['valid'][metric_key])
            tst.append(el['test'][metric_key])

        x_str = '#Suppervised samples'
        y_str = "{} (%)".format(metric_tag)
        knn, certain = None, None
        lambdx = None
        al_type = lstats[0]['propagation']['args']['al_type']
        clustering = lstats[0]['propagation']['args']['clustering']
        dataset = lstats[0]['propagation']['args']['dataset']
        ours = (al_type == constants.AL_LP)
        fixed = (al_type in [constants.AL_FULL_SUP, constants.AL_WSL])

        if ours:
            knn = lstats[0]['propagation']['args']['knn']
            lambdx = lstats[0]['propagation']['args']['scale_seg_u']

        xticks = [format(v, '.0e') if v >= 1000 else str(v) for v in nbr_sup]

        for i in range(len(nbr_sup)):
            if (i % self.frequency) != 0:
                xticks[i] = ''

        for tag, val in zip(['train', 'valid', 'test'], [tr, vl, tst]):
            if fixed:
                val = [val[0] for _ in range(self.max_rep)]
                xticks = ['' for _ in range(self.max_rep)]
                xticks[int(self.max_rep / 2)] = format(nbr_sup[0], '.0e')

            path = join(self.current_folder_rounds, 'vision',
                        '{}-{}.png'.format(tag, metric_key)
                        )
            title = '{} {}set ({}). Method: {}.'.format(
                metric_tag, tag, dataset, al_type)
            if ours:
                title = '{} knn: {}. Lambda: {}, Clustering: {}.'.format(
                    title, knn, lambdx, clustering)

            self.plot_curve(xticks=xticks,
                            values=val,
                            path=path,
                            title=title,
                            x_str=x_str,
                            y_str=y_str,
                            ylimtop=ylimtop,
                            ylimbottom=ylimbottom
                            )

        gold = {
            'nbr_sup': nbr_sup,
            'train': tr,
            'valid': vl,
            'test': tst
        }
        return gold

    def plot_metric_pseudo_l(self,
                             lstats,
                             metric_key,
                             metric_tag
                             ):
        """
        Plot stats over the pseudo labeled samples. this Concerns only our
        method.
        """
        nbr_sup, nbr_pseudol = [], []

        perf_pseudo_knn = []  # cl task

        # pseyudo-labeled samples. +++++++++++++++++++++++++++++++++++++++++++++
        avg_dice_final_pl = []  # seg task. avg_dice per-al-round of
        # all pairs at the final cycle (cycle=1).  [final]
        sum_dices_pl_final = 0.  # seg. task. contains the sum of dice of all
        # paired samples at the cycle=1. [final]
        sum_nbr_s_np_final = 0.  # seg.task. contains the sum of number of all
        # pairs at cycle=1. [final]

        avg_dice_init_pl = []  # seg task. avg_dice per-al-round of
        # newly paired samples at the first cycle (cycle=0).
        sum_dices_pl_init = 0.  # seg. task. contains the sum of dice of all
        # newly paired samples at the cycle=0.
        sum_nbr_s_np_init = 0.  # seg.task. contains the sum of number of pairs
        # newly paired at cycle=0


        # leftovers ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        perf_leftovers = []  # segm.task.

        perf_pseudo_pred = []  # cl task
        al_type, dataset = None, None

        for el in lstats:
            al_type = el['propagation']['args']['al_type']
            assert al_type == constants.AL_LP, "something is wrong."

            nbr_sup.append(el['propagation']['nbr_fully_sup'])
            nbr_pseudol.append(el['propagation']['nbr_samples_prop'])

            if self.task == constants.CL:
                perf_pseudo_knn.append(el['propagation'][metric_key])
                perf_pseudo_pred.append(
                    el['trainset-pseudoL'][metric_key] * 100.)
            elif self.task == constants.SEG:
                 # pseudo-labeled -----------------------------------------------
                # init.
                # computes the true average over the ALL samples that
                # have been NEWLY paired over all AL-ROUNDs: one scalar.
                if 'number_new_p' in el['propagation'].keys():
                    nbr_s_new_p = float(el['propagation']['number_new_p'])
                    sum_dices_pl_init += el['propagation']['sum_dice_new_p']
                    sum_nbr_s_np_init +=  nbr_s_new_p

                    if nbr_s_new_p > 0:
                        avg_dice_init_pl.append(
                            el['propagation']['sum_dice_new_p'] / nbr_s_new_p
                        )
                    else:
                        avg_dice_init_pl.append(0.)

                # final.
                # computes the true average over the ALL samples that
                # have been paired over all AL-ROUNDs: one scalar.
                if 'number_all_pairs' in el['propagation'].keys():
                    nbr_s_all_p = float(el['propagation']['number_all_pairs'])
                    sum_dices_pl_final += el['propagation'][
                        'sum_dice_all_pairs']
                    sum_nbr_s_np_final += nbr_s_all_p

                # final. +++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # values are in [0, 100]
                if el['trainset-pseudoL'] is not None:
                    avg_dice_final_pl.append(el['trainset-pseudoL'][metric_key])
                else:
                    avg_dice_final_pl.append(0.)
                # leftovers ----------------------------------------------------
                # values are in [0, 100]
                if el['trainset-leftovers'] is not None:
                    perf_leftovers.append(el['trainset-leftovers'][metric_key])
                else:
                    perf_leftovers.append(0.)

        true_avg_dice_init_pl = 0.
        true_avg_dice_final_pl = 0.
        if (self.task == constants.SEG) and (sum_nbr_s_np_init > 0):
            true_avg_dice_init_pl = sum_dices_pl_init / sum_nbr_s_np_init

        if (self.task == constants.SEG) and (sum_nbr_s_np_final > 0):
            true_avg_dice_final_pl = sum_dices_pl_final / sum_nbr_s_np_final



        x_str = 'Active learning rounds'
        knn = lstats[0]['propagation']['args']['knn']
        lambdx = lstats[0]['propagation']['args']['scale_seg_u']
        clustering = lstats[0]['propagation']['args']['clustering']
        dataset = lstats[0]['propagation']['args']['dataset']
        font_sz = 6
        x = list(range(len(nbr_sup)))
        xticks = [str(v) for v in x]

        # ======================================================================
        #                   NUMBER OF SAMPLES: SUP VS PSEUDO-LABELED
        #                               TASKS: CL, SEG
        # ======================================================================
        fig = plt.figure()

        plt.plot(x, nbr_pseudol, label='#pseudo-labeled samples',
                 linestyle='--')
        plt.plot(x, nbr_sup, label='#sup samples', linestyle='-')

        title = ''
        if self.task == constants.CL:
            title= '#sup vs #pseudo-lab. Dataset: {}. ' \
                   'Method: {}. knn: {}. Clustering: {}.'.format(
                dataset, al_type, knn, clustering)
        elif self.task == constants.SEG:
            title = '#sup vs #pseudo-lab. Dataset: {}. ' \
                    'Method: {}. knn: {}. Lambda: {}.' \
                    'Clustering: {}.'.format(
                dataset, al_type, knn, lambdx, clustering)

        path = join(self.current_folder_rounds,
                    'vision',
                    '{}.png'.format('sample_sup-vs-pseudo-labeled.png'))

        self.decorate_figure(fig,
                             path,
                             title=title,
                             x_str=x_str,
                             y_str='#samples',
                             dpi=300,
                             ylimtop=None,
                             ylimbottom=None,
                             legend_loc="lower right",
                             xticks=xticks,
                             x=x
                             )

        # ======================================================================
        #                 -  AVERAGE DICE (INIT.FINAL)
        #                  OVER PSEUDO-LABELED SAMPLES
        #                 -  AVERAGE DICE OVER LEFTOVERS SAMPLES
        #                       TASKS: SEG.
        # ======================================================================

        # 1. initial.dice of newly paired samples.
        avg_dice_init_pl = np.array(avg_dice_init_pl) * 100.
        if (self.task == constants.SEG) and (avg_dice_init_pl.size > 0):
            title = 'Average Dice index over newly-paired samples (cycle 0). ' \
                    '(AVG/AL-ROUND). knn: {}. Lamb: {}'.format(knn, lambdx)
            path = join(self.current_folder_rounds,
                        'vision',
                        '{}.png'.format(
                            'avg-dice-idx-init-paired-samples-cycle-0.png'))

            self.plot_curve(xticks=xticks,
                            values=avg_dice_init_pl,
                            path=path,
                            title=title,
                            x_str=x_str,
                            y_str='Dice index (%)',
                            dpi=300,
                            ylimtop=100.,
                            ylimbottom=0.,
                            legend_loc="lower right",
                            x=x
                            )

        # 2. final.dice of ALL pairs (new + previous)
        avg_dice_final_pl = np.array(avg_dice_final_pl)
        if (self.task == constants.SEG) and (avg_dice_final_pl.size > 0):
            title = 'Average Dice index over all-paired samples (cycle 1). ' \
                    '(AVG/AL-ROUND). knn: {}. Lamb: {}'.format(knn, lambdx)
            path = join(self.current_folder_rounds,
                        'vision',
                        '{}.png'.format(
                            'avg-dice-idx-all-paired-samples-cycle-1.png'))

            self.plot_curve(xticks=xticks,
                            values=avg_dice_final_pl,
                            path=path,
                            title=title,
                            x_str=x_str,
                            y_str='Dice index (%)',
                            dpi=300,
                            ylimtop=100.,
                            ylimbottom=0.,
                            legend_loc="lower right",
                            x=x
                            )

        # 3. leftovers.
        perf_leftovers = np.array(perf_leftovers)
        if (self.task == constants.SEG) and (perf_leftovers.size > 0):

            title = 'Average Dice index over leftover samples. ' \
                    'knn: {}. Lamb: {}'.format(knn, lambdx)
            path = join(self.current_folder_rounds,
                        'vision',
                        '{}.png'.format(
                            'avg-dice-idx-leftover-samples.png'))

            self.plot_curve(xticks=xticks,
                            values=perf_leftovers,
                            path=path,
                            title=title,
                            x_str=x_str,
                            y_str='Dice index (%)',
                            dpi=300,
                            ylimtop=100.,
                            ylimbottom=0.,
                            legend_loc="lower right",
                            x=x
                            )

        # ======================================================================
        #          PERFORMANCE OVER PSEUDO-LABELED SAMPLES: KNN VS PREDICTION
        #                     TASKS: CL.
        # ======================================================================
        fig = plt.figure()

        if self.task == constants.CL:
            plt.plot(x, perf_pseudo_pred,
                     label='Network prediction (after training)',
                     linestyle='--')

            plt.plot(x, perf_pseudo_knn, label='knn (k={})'.format(knn),
                     linestyle='-')
            plt.title('{} over pseudo-labeled samples. Dataset: {}, '
                      'Method: {}. knn: {}. Clustering: {}.'.format(
                metric_tag, dataset, al_type, knn, clustering),
                fontsize=font_sz)

            title = '{} over pseudo-labeled samples. Dataset: {},' \
                    'Method: {}. knn: {}. Clustering: {}.'.format(
                metric_tag, dataset, al_type, knn, clustering)

            path = join(self.current_folder_rounds,
                        'vision',
                        '{}.png'.format('perf_knn_vs_net_prediction.png'))

            self.decorate_figure(fig,
                                 path,
                                 title=title,
                                 x_str=x_str,
                                 y_str='{}%'.format(metric_tag),
                                 dpi=300,
                                 ylimtop=100.,
                                 ylimbottom=0.,
                                 legend_loc="lower right",
                                 xticks=xticks,
                                 x=x
                                 )


        gold = {
            'nbr_sup': nbr_sup,  # cl, seg
            'nbr_pseudol': nbr_pseudol,  # cl, seg
            'perf_peudo_knn': perf_pseudo_knn,  # cl
            'perf_peudo_pred': perf_pseudo_pred,  # cl
            'avg_dice_final_pl': avg_dice_final_pl,  # seg
            'avg_dice_init_pl': avg_dice_init_pl,  #seg
            'perf_leftovers': perf_leftovers,  # seg (only!)
            'true_avg_dice_init_pl': true_avg_dice_init_pl,  # seg
            'true_avg_dice_final_pl': true_avg_dice_final_pl  # seg
        }
        return gold

    def __call__(self,
                 fd=None
                 ):
        """
        Plot.
        :param fd: str for the folder containing all the rounds.
        if None, we use self.folder_rounds.
        """
        if fd is None:
            fd = self.folder_rounds

        self.current_folder_rounds = fd

        # find all folders of al rounds.
        list_rds = []
        for d in os.listdir(fd):
            if d.startswith('mth') and os.path.isdir(join(fd, d)):
                list_rds.append(join(fd, d))

        # ======================================================================
        #                      START: GATHER STATS
        # ======================================================================

        nbr_it = len(list_rds)
        lstats_it = [None for _ in range(nbr_it)]
        al_lp = False
        for fd in list_rds:
            stats_it = dict()
            # read propagation.pkl
            with open(join(fd, 'propagation.pkl'), 'rb') as fpr:
                prop = pkl.load(fpr)

            stats_it['propagation'] = deepcopy(prop)
            al_type = prop['args']['al_type']

            # get train, valid, test stats.
            for k in ['train', 'valid', 'test']:
                fpkl = join(fd, k, 'final-tracker-{}set.pkl'.format(k))
                with open(fpkl, 'rb') as ftr:
                    statsset = pkl.load(ftr)
                stats_it[k] = deepcopy(statsset)

            if al_type == constants.AL_LP:
                al_lp = True
                # 1. train-pseudo-labeled samples.
                fpkl = join(fd,
                            'train',
                            'final-tracker-trainset-pseudoL-True.pkl'
                            )
                if os.path.exists(fpkl):
                    with open(fpkl, 'rb') as ftr:
                        statsset = pkl.load(ftr)

                    stats_it['trainset-pseudoL'] = deepcopy(statsset)
                else:
                    stats_it['trainset-pseudoL'] = None

                # 2. leftovers.
                fpkl = join(fd,
                            'leftovers',
                            'final-tracker-trainset-leftovers.pkl'
                            )
                if os.path.exists(fpkl):
                    with open(fpkl, 'rb') as ftr:
                        statsset = pkl.load(ftr)

                    stats_it['trainset-leftovers'] = deepcopy(statsset)
                else:
                    stats_it['trainset-leftovers'] = None

            lstats_it[int(prop['args']['al_it'])] = stats_it

        # ======================================================================
        #                      END: GATHER STATS
        # ======================================================================

        # ======================================================================
        #                      START: PLOT STATS
        # ======================================================================
        acc_key = 'acc'
        acc_tag = 'Classification accuracy'
        dice_key = 'dice_idx'
        dice_tag = 'Dice index'

        gold_acc = self.plot_metric(lstats_it,
                                    metric_key=acc_key,
                                    metric_tag=acc_tag,
                                    ylimtop=100.,
                                    ylimbottom=0.
                                    )
        gold_dice_idx = self.plot_metric(lstats_it,
                                         metric_key=dice_key,
                                         metric_tag=dice_tag,
                                         ylimtop=100.,
                                         ylimbottom=0.
                                         )
        gold_acc_pseudo_l = None  # classification accuracy image level
        gold_dice_pseudo_l = None  # Dice index.

        if al_lp and (self.task == constants.CL):  # it does not matter for SEG.
            gold_acc_pseudo_l = self.plot_metric_pseudo_l(lstats_it,
                                                          metric_key=acc_key,
                                                          metric_tag=acc_tag
                                                          )
        if al_lp and (self.task == constants.SEG):
            gold_dice_pseudo_l = self.plot_metric_pseudo_l(lstats_it,
                                                           metric_key=dice_key,
                                                           metric_tag=dice_tag
                                                           )
        # ======================================================================
        #                      END: PLOT STATS
        # ======================================================================

        gold = {
            'lstats_it': lstats_it,
            "acc": gold_acc,
            'dice_idx': gold_dice_idx,
            'pseudo_l_cc': gold_acc_pseudo_l,
            'pseudo_l_dice': gold_dice_pseudo_l,
            'task': self.task
        }

        path = join(self.current_folder_rounds, 'vision', 'gold.pkl')
        with open(path, 'wb') as fout:
            pkl.dump(gold, fout, protocol=pkl.HIGHEST_PROTOCOL)



