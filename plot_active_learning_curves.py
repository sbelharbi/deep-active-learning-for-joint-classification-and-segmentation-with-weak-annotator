"""
Plot the curves of all the active learning method of a dataset.
Store the figure in the folder `paper`.
"""
import os
import sys
import fnmatch
from os.path import expanduser
from os.path import join
import copy


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm
import pickle as pkl
import numpy as np
from texttable import Texttable
import csv

import constants
from vision import PlotActiveLearningRounds
from shared import announce_msg
from shared import compute_auc

mpl.style.use('seaborn')
# styles:
# https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html
# https://matplotlib.org/3.3.0/gallery/style_sheets/style_sheets_reference.html

# choose the datset
dataset = constants.CUB

announce_msg("Processing dataset: {}".format(dataset))

show_percentage = True  # if true, the xlabels will be percentage instead of
# number of samples..

TAG = 'paper_label_prop/{}'.format(dataset)

# colors: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
colors = {
    constants.AL_FULL_SUP: mcolors.CSS4_COLORS['black'],
    constants.AL_LP: mcolors.CSS4_COLORS['red'],
    constants.AL_ENTROPY: mcolors.CSS4_COLORS['sienna'],
    constants.AL_RANDOM: mcolors.CSS4_COLORS['blue'],
    constants.AL_MCDROPOUT: mcolors.CSS4_COLORS['green'],
    constants.AL_WSL: mcolors.CSS4_COLORS['orange']
}

limits = {
    constants.GLAS: {
        "up": 89.,
        "down": 64.,
        "step": 4.
    },
    constants.CUB: {
        "up": 82.,
        "down": 35.,
        "step": 10.
    },
    constants.OXF: {
        "up": 89.,
        "down": 40.,
        "step": 10.
    }
}

order_plot = [
    constants.AL_WSL,
    constants.AL_RANDOM,
    constants.AL_ENTROPY,
    constants.AL_MCDROPOUT,
    constants.AL_LP,
    constants.AL_FULL_SUP
]

# true name of the dataset.
datasetname = {
    constants.GLAS: "GlaS",
    constants.CUB: "CUB",
    constants.OXF: "OXF"
}

init_acc = {
    constants.GLAS: 92.50,
    constants.CUB: 73.76,
    constants.OXF: 82.03
}

# folder where all the exps of dataset live.
fd_in = join('./exps/{}'.format(TAG))


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), "Folder {} does not exist " \
                                   ".... [NOT OK]".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def plot_curves(xlabels,
                lvalues,
                legends,
                path,
                title="",
                x_str="",
                y_str="",
                dpi=300,
                ylim=None
                ):
    """
    Plot a curve.

    :param xlabels: list of x ticks labels.
    :param lvalues: list of lists or numpy.ndarray of values to plot (y)
    :param legends: list of legends.
    :param path: str, path where to save the figure.
    :param title: str, the title of the plot.
    :param x_str: str, the name of the x axis.
    :param y_str: str, the name of the y axis.
    :param dpi: int, the dpi of the image.
    """

    font_sz = 5

    fig = plt.figure()
    x = list(range(len(lvalues[0])))
    for vl, leg in zip(lvalues, legends):
        x = list(range(len(vl)))
        plt.plot(x, vl, label=leg)
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.legend(loc="lower right", prop={'size': 5})
    plt.title(title, fontsize=font_sz)
    plt.xticks(x, xlabels, fontsize=5)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    if ylim is not None:
        plt.ylim(0, 105)
    del fig


def dump_iterable_in_txt_file(fdump, itera, head, nbr):
    """
    Dump an iterable into text file
    :param fdump: object of text file opened.
    :param itera: iterable through one index.
    :param head: str. text contains what represents the data in iterbale.
    :param nbr: int. number of elements in iterable.
    :return:
    """
    isstr = isinstance(itera[0], str)
    for kk in range(nbr):
        if kk == 0:
            if isstr:
                fdump.write("{}: {}, ".format(head, itera[kk]))
            else:
                fdump.write("{}: {:.3f}, ".format(head, itera[kk]))
        elif kk < (nbr - 1):
            if isstr:
                fdump.write("{}, ".format(itera[kk]))
            else:
                fdump.write("{:.3f}, ".format(itera[kk]))
        else:
            if isstr:
                fdump.write("{} \n".format(itera[kk]))
            else:
                fdump.write("{:.3f} \n".format(itera[kk]))


def overlap_curves(fig,
                   xlabels,
                   avg,
                   std,
                   legend,
                   color,
                   path,
                   title="",
                   x_str="",
                   y_str="",
                   dpi=300,
                   ylimup=None,
                   ylimdown=None,
                   step=10.
                   ):
    """
    Plot a curve.

    :param xlabels: list of x ticks labels.
    :param lvalues: list of lists or numpy.ndarray of values to plot (y)
    :param legends: list of legends.
    :param path: str, path where to save the figure.
    :param title: str, the title of the plot.
    :param x_str: str, the name of the x axis.
    :param y_str: str, the name of the y axis.
    :param dpi: int, the dpi of the image.
    """

    if ylimup is None:
        ylimup = 105.

    if ylimdown is None:
        ylimdown = 0.

    font_sz = 10
    tiks_fsz = 7
    plt.figure(fig.number)
    x = list(range(avg.size))
    linewidth = 1.5
    plt.plot(x, avg, label=legend, color=color, linewidth=linewidth)
    plt.fill_between(x, avg - std, avg + std, alpha=0.1, color=color)
    plt.xlabel(x_str, fontsize=font_sz)
    plt.ylabel(y_str, fontsize=font_sz)

    plt.legend(loc="lower right", prop={'size': 10})
    plt.title(title, fontsize=font_sz)
    plt.xticks(x, xlabels, fontsize=tiks_fsz)

    ylabels = [str(i) for i in range(int(ylimdown), int(ylimup), int(step))]
    y = list(range(int(ylimdown), int(ylimup), int(step)))
    plt.yticks(y, ylabels, fontsize=tiks_fsz)

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.1)

    # Show the minor grid lines with very faint and almost transparent grey
    # lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)

    plt.ylim(ylimdown, ylimup)
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
    # plt.close('all')


    return fig


def writetab(llargs, fout, kset):
    t = Texttable()
    t.set_max_width(500)
    info = ['dataset', 'al_type', 'knn', 'clustering']
    restricted = ['knn', 'clustering']
    header = []
    format_row = []

    # if show_exp_fd:
    #     header = ["exp_fd"]
    #     format_row.append("t")
    for k in info:
        if k not in ["exp_fd"]:
            header.append(k)
            format_row.append("t")  # text

    header.append('AUC*')
    format_row.append('f')
    t.header(header)
    t.set_cols_dtype(format_row)

    for item, aucx in llargs:
        row = []
        for k in info:
            if item['al_type'] == constants.AL_LP:
                row.append(item[k])
            else:
                if k in restricted:
                    row.append(None)
                else:
                    row.append(item[k])

        row.append(aucx)
        t.add_row(row)

    print("Subset: {}".format(kset))
    print(t.draw())  # print in std_out
    ff = open(fout, 'w')
    print(t.draw(), file=ff)
    print("Subset: {}".format(kset), file=ff)
    ff.close()


def scan_all():
    """
    plot average curves of all methods in folder "fd_in".
    """
    lgold = find_files_pattern(fd_in, 'gold.pkl')
    print("Found {} gold files.".format(len(lgold)))
    subsets = ['test']  # ['train', 'valid', 'test']
    methods = dict()
    full_sz = None
    nbr_sup = None

    # find all golds
    for gold in tqdm.tqdm(lgold, ncols=80, total=len(lgold)):
        with open(gold, 'rb') as fin:
            stats = pkl.load(fin)

        # fix full sup
        # args = stats['lstats_it'][0]['propagation']['args']
        # if args['al_type'] == constants.AL_FULL_SUP:
        #     fdxx = "/".join(gold.split("/")[:-2])
        #     print("fixing {}".format(fdxx))
        #     plotter = PlotActiveLearningRounds(folder_rounds=fdxx)
        #     plotter()
        #
        # continue
        #
        # sys.exit()

        args = stats['lstats_it'][0]['propagation']['args']
        al_type = args["al_type"]
        ds = args["dataset"]
        assert ds == dataset, "found dataset {} that is different from the " \
                              "config_dataset {}".format(ds, dataset)

        if args['al_type'] not in [constants.AL_FULL_SUP, constants.AL_WSL]:
            nbr_sup = stats['acc']['nbr_sup']
            if not show_percentage:
                xlabels = [str(i) for i in nbr_sup]
            elif full_sz is not None:
                xlabels = [
                    "{:.0f}".format(100. * i/float(full_sz)) for i in nbr_sup]
        else:
            full_sz = stats['acc']['nbr_sup'][0]

        if full_sz is not None and nbr_sup is not None:
            xlabels_perc = [
                "{:.0f}".format(100. * i / float(full_sz)) for i in nbr_sup]

        if nbr_sup is not None:
            xlabels_nbr_s = [str(i) for i in nbr_sup]

        if args['al_type'] == constants.AL_LP:
            knn = args["knn"]

        if al_type in methods:
            methods[al_type]["l_stats"].append(stats)
        else:
            methods[al_type] = dict()
            methods[al_type]["l_stats"] = [stats]
            methods[al_type]["dataset"] = ds

    # compute the mean and std_dev of each method, then plot them.
    figs_acc = dict()
    figs_di = dict()
    for kset in subsets:
        figs_acc[kset] = plt.figure()
        figs_di[kset] = plt.figure()

    i = 0
    output_fd = "paper"
    if not os.path.exists(output_fd):
        os.makedirs(output_fd)

    fout = open(join(output_fd, "{}.txt".format(dataset)), 'w')
    fcsv = open(join(output_fd, "{}.csv".format(dataset)), 'w')
    csvw = csv.writer(fcsv)
    csvw.writerow(["Dataset: {}".format(dataset)])
    csvw.writerow(["NBR-SUP"])
    csvw.writerow(xlabels_nbr_s)

    csvw.writerow(['NBR-SUP (%)'])
    csvw.writerow(xlabels_perc)

    fout.write("Dataset: {} \n".format(dataset))

    fout.write("Total samples: {} \n".format(full_sz))
    fout.write("Init-CL-ACC: {} %\n".format(init_acc[dataset]))

    # plot curves following some order.
    keys_methds = [key for key in order_plot if key in list(methods.keys())]

    for al_type in keys_methds:
        l_stats = methods[al_type]["l_stats"]
        fout.write("=" * 100 + '\n')
        fout.write("Method {}: \n".format(al_type))

        # compute the mean
        for kset in subsets:
            fout.write("Subset: {} \n".format(kset))

            tmp_stats = {
                "acc": [],
                "dice_idx": []
            }
            tmp_auc = {
                "acc": [],
                "dice_idx": []
            }

            # loop over seeds
            for seed_stats in l_stats:
                # acc
                tmp_stats['acc'].append(seed_stats["acc"][kset])
                tmp_stats['dice_idx'].append(seed_stats["dice_idx"][kset])
                tmp_auc['acc'].append(
                    compute_auc(np.array(seed_stats["acc"][kset]), len(nbr_sup))
                )
                tmp_auc['dice_idx'].append(
                    compute_auc(np.array(seed_stats["dice_idx"][kset]),
                                len(nbr_sup))
                )
                args = stats['lstats_it'][0]['propagation']['args']

                if len(seed_stats['acc'][kset]) == 19:
                    print(args['MYSEED'], al_type, kset)

            sz = len(tmp_stats['acc'][0])
            for xxx in tmp_stats['acc']:
                assert len(xxx) == sz, "different sizes acc: {} {}".format(
                    sz, len(xxx))

            szdi = len(tmp_stats['dice_idx'][0])
            for xxx in tmp_stats['dice_idx']:
                assert len(xxx) == szdi, "different sizes dice: {} {}".format(
                    szdi, len(xxx))

            tmp_stats['acc'] = np.array(tmp_stats['acc'])
            tmp_stats['dice_idx'] = np.array(tmp_stats['dice_idx'])
            tmp_auc['acc'] = np.array(tmp_auc['acc'])
            tmp_auc['dice_idx'] = np.array(tmp_auc['dice_idx'])

            print(tmp_stats['acc'].shape, al_type, kset)
            print(tmp_stats['dice_idx'].shape, al_type, kset)

            avg_acc = tmp_stats['acc'].mean(axis=0)
            std_acc = tmp_stats['acc'].std(axis=0)
            avg_di = tmp_stats['dice_idx'].mean(axis=0)
            std_di = tmp_stats['dice_idx'].std(axis=0)
            methods[al_type][kset] = dict()
            methods[al_type][kset]["avg_acc"] = avg_acc
            methods[al_type][kset]["std_acc"] = std_acc
            methods[al_type][kset]["avg_di"] = avg_di
            methods[al_type][kset]["std_di"] = std_di

            if al_type in [constants.AL_FULL_SUP, constants.AL_WSL]:
                avg_acc = np.array(
                    [avg_acc for _ in range(len(nbr_sup))]).squeeze()
                std_acc = np.array(
                    [std_acc for _ in range(len(nbr_sup))]).squeeze()

                avg_di = np.array(
                    [avg_di for _ in range(len(nbr_sup))]).squeeze()
                std_di = np.array(
                    [std_di for _ in range(len(nbr_sup))]).squeeze()
            # plot
            # acc ==============================================================
            avg_area_acc = tmp_auc['acc'].mean()
            std_area_acc = tmp_auc['acc'].std()

            if al_type != constants.AL_LP:
                legend = r"{} (AUC: {:.2f}$\pm${:.2f})".format(
                    al_type, avg_area_acc, std_area_acc)
            else:
                legend = r"{} [$k$: {}] (AUC: {:.2f}$\pm${:.2f})".format(
                    al_type, knn, avg_area_acc, std_area_acc)

            title = "Classification accuracy. Dataset: {} ({}).".format(
                datasetname[dataset], kset)
            fname = "{}-cl-acc-{}.png".format(dataset, kset)

            fout.write("AUC acc: {} +- {} \n".format(
                avg_area_acc, std_area_acc))

            csvw.writerow(["Method: {}".format(al_type)])
            csvw.writerow(["AVG_ACC:"])
            csvw.writerow(avg_acc)
            csvw.writerow(["STD_ACC:"])
            csvw.writerow(std_acc)

            if show_percentage:
                x_str = "Percentage of labeled samples (%)"
            else:
                x_str = "#supervised samples"
            y_str = 'Classification accuracy (%)'

            tmp_xlbales = copy.deepcopy(xlabels)
            tmp_xlbales.insert(0, str(0))
            figs_acc[kset] = overlap_curves(
                fig=figs_acc[kset],
                xlabels=tmp_xlbales,
                avg=np.hstack((np.array(init_acc[dataset]), avg_acc)),
                std=np.hstack((np.array([0.0]), std_acc)),
                legend=legend, color=colors[al_type],
                path=join(output_fd, fname), title=title,
                x_str=x_str,
                y_str=y_str,
                dpi=200,
                ylimup=105,
                ylimdown=0.,
                step=10.
            )

            # dice =============================================================
            avg_area_di = tmp_auc['dice_idx'].mean()
            std_area_di = tmp_auc['dice_idx'].std()

            if al_type != constants.AL_LP:
                legend = r"{} (AUC: {:.2f}$\pm${:.2f})".format(
                    al_type, avg_area_di, std_area_di)
            else:
                legend = r"{} [$k$: {}] (AUC: {:.2f}$\pm${:.2f})".format(
                    al_type, knn, avg_area_di, std_area_di)

            title = "Dice index. Dataset: {} ({}).".format(
                datasetname[dataset], kset)
            fname = "{}-dice-idx-{}.png".format(dataset, kset)

            fout.write("AUC Dice index: {} +- {} \n".format(
                avg_area_di, std_area_di))

            csvw.writerow(["AVG_DICE:"])
            csvw.writerow(avg_di)
            csvw.writerow(["STD_DICE:"])
            csvw.writerow(std_di)

            if show_percentage:
                x_str = "Percentage of labeled samples (%)"
            else:
                x_str = "#supervised samples"
            y_str = 'Dice Index (%)'

            figs_di[kset] = overlap_curves(
                fig=figs_di[kset],
                xlabels=xlabels,
                avg=avg_di,
                std=std_di,
                legend=legend, color=colors[al_type],
                path=join(output_fd, fname), title=title,
                x_str=x_str,
                y_str=y_str,
                dpi=200,
                ylimup=limits[dataset]['up'],
                ylimdown=limits[dataset]['down'],
                step=limits[dataset]['step']
            )

    fout.close()
    fcsv.close()


if __name__ == "__main__":
    scan_all()






