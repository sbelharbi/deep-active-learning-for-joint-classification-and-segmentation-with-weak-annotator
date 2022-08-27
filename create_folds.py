"""
Splits the following dataset into k-folds for active learning:
1. GlaS.
2. Caltech-UCSD-Birds-200-2011
3. Oxford_flowers_102
"""

from os.path import join, relpath, basename, splitext, isfile
import os
import random
import sys
import math
import csv
import copy
import fnmatch

import yaml
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageChops
import tqdm
import matplotlib.pyplot as plt


from tools import chunk_it
from tools import Dict2Obj
from tools import get_rootpath_2_dataset

from shared import announce_msg, check_if_allow_multgpu_mode
from shared import csv_loader

import constants


import reproducibility


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


def show_msg(ms, lg):
    announce_msg(ms)
    lg.write(ms + "\n")


def get_stats(args, split, fold, subset):
    """
    Get some stats on the image sizes of specific dataset, split, fold.
    """
    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    tag = "ds-{}-s-{}-f-{}-subset-{}".format(
        args.dataset, split, fold, subset
    )
    log = open(join(
        args.fold_folder, "log-stats-ds-{}.txt".format(tag)), 'w')
    announce_msg("Going to check {}".format(args.dataset.upper()))

    relative_fold_path = join(
        args.fold_folder, "split_{}".format(split), "fold_{}".format(fold)
    )

    subset_csv = join(relative_fold_path, "{}_s_{}_f_{}.csv".format(
                    subset, split, fold))
    rootpath = get_rootpath_2_dataset(args)
    samples = csv_loader(subset_csv, rootpath)

    lh, lw = [], []
    for el in samples:
        img = Image.open(el[1], 'r').convert('RGB')
        w, h = img.size
        lh.append(h)
        lw.append(w)

    msg = "min h {}, \t max h {}".format(min(lh), max(lh))
    show_msg(msg, log)
    msg = "min w {}, \t max w {}".format(min(lw), max(lw))
    show_msg(msg, log)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].hist(lh)
    axes[0].set_title('Heights')
    axes[1].hist(lw)
    axes[1].set_title('Widths')
    fig.tight_layout()
    plt.savefig(join(args.fold_folder, "size-stats-{}.png".format(tag)))

    log.close()


def dump_fold_into_csv_glas(lsamples, outpath, tag):
    """
    For glas dataset.

    Write a list of list of information about each sample into a csv
    file where each row represent a sample in the following format:
    row = "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4
    Possible tags:
    0: labeled
    1: unlabeled
    2: labeled but came from unlabeled set. [not possible at this level]

    Relative paths allow running the code an any device.
    The absolute path within the device will be determined at the running
    time.

    :param lsamples: list of tuple (str: relative path to the image,
    float: id, str: label).
    :param outpath: str, output file name.
    :param tag: int, tag in constants.samples_tags for ALL the samples.
    :return:
    """
    msg = "'tag' must be an integer. Found {}.".format(tag)
    assert isinstance(tag, int), msg
    msg = "'tag' = {} is unknown. Please see " \
          "constants.samples_tags = {}.".format(tag, constants.samples_tags)
    assert tag in constants.samples_tags, msg

    msg = "It is weird that you are tagging a sample as {} while we" \
          "are still in the splitting phase. This is not expected." \
          "We're out.".format(constants.PL)
    assert tag != constants.PL, msg

    with open(outpath, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, clas, id_ in lsamples:
            filewriter.writerow(
                [str(int(id_)),
                 name + ".bmp",
                 name + "_anno.bmp",
                 clas,
                 tag]
            )


def dump_fold_into_csv_CUB(lsamples, outpath, tag):
    """
    for Caltech_UCSD_Birds_200_2011 dataset.
    Write a list of list of information about each sample into a csv
    file where each row represent a sample in the following format:
    row = "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4
    Possible tags:
    0: labeled
    1: unlabeled
    2: labeled but came from unlabeled set. [not possible at this level]

    Relative paths allow running the code an any device.
    The absolute path within the device will be determined at the running
    time.

    :param lsamples: list of tuple (str: relative path to the image,
    float: id, str: label).
    :param outpath: str, output file name.
    :param tag: int, tag in constants.samples_tags for ALL the samples.
    :return:
    """
    msg = "'tag' must be an integer. Found {}.".format(tag)
    assert isinstance(tag, int), msg
    msg = "'tag' = {} is unknown. Please see " \
          "constants.samples_tags = {}.".format(tag, constants.samples_tags)
    assert tag in constants.samples_tags, msg

    msg = "It is weird that you are tagging a sample as {} while we" \
          "are still in the splitting phase. This is not expected." \
          "We're out.".format(constants.PL)
    assert tag != constants.PL, msg

    with open(outpath, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img_path, mask_path, img_label, idcnt in lsamples:
            filewriter.writerow(
                [str(int(idcnt)),
                 img_path,
                 mask_path,
                 img_label,
                 tag]
            )


def dump_fold_into_csv_OXF(lsamples, outpath, tag):
    """
    for Oxford_flowers_102 dataset.
    Write a list of list of information about each sample into a csv
    file where each row represent a sample in the following format:
    row = "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4
    Possible tags:
    0: labeled
    1: unlabeled
    2: labeled but came from unlabeled set. [not possible at this level]

    Relative paths allow running the code an any device.
    The absolute path within the device will be determined at the running
    time.

    :param lsamples: list of tuple (str: relative path to the image,
    float: id, str: label).
    :param outpath: str, output file name.
    :param tag: int, tag in constants.samples_tags for ALL the samples.
    :return:
    """
    msg = "'tag' must be an integer. Found {}.".format(tag)
    assert isinstance(tag, int), msg
    msg = "'tag' = {} is unknown. Please see " \
          "constants.samples_tags = {}.".format(tag, constants.samples_tags)
    assert tag in constants.samples_tags, msg

    msg = "It is weird that you are tagging a sample as {} while we" \
          "are still in the splitting phase. This is not expected." \
          "We're out.".format(constants.PL)
    assert tag != constants.PL, msg

    with open(outpath, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img_path, mask_path, img_label, idcnt in lsamples:
            filewriter.writerow(
                [str(int(idcnt)),
                 img_path,
                 mask_path,
                 img_label,
                 tag]
            )


def dump_fold_into_csv_CAM16(lsamples, outpath, tag):
    """
    for camelyon16 dataset.
    Write a list of list of information about each sample into a csv
    file where each row represent a sample in the following format:
    row = "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4
    Possible tags:
    0: labeled
    1: unlabeled
    2: labeled but came from unlabeled set. [not possible at this level]

    Relative paths allow running the code an any device.
    The absolute path within the device will be determined at the running
    time.

    :param lsamples: list of tuple (str: relative path to the image,
    float: id, str: label).
    :param outpath: str, output file name.
    :param tag: int, tag in constants.samples_tags for ALL the samples.
    :return:
    """
    msg = "'tag' must be an integer. Found {}.".format(tag)
    assert isinstance(tag, int), msg
    msg = "'tag' = {} is unknown. Please see " \
          "constants.samples_tags = {}.".format(tag, constants.samples_tags)
    assert tag in constants.samples_tags, msg

    msg = "It is weird that you are tagging a sample as {} while we" \
          "are still in the splitting phase. This is not expected." \
          "We're out.".format(constants.PL)
    assert tag != constants.PL, msg

    with open(outpath, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for idcnt, img_path, mask_path, img_label in lsamples:
            filewriter.writerow(
                [str(int(idcnt)),
                 img_path,
                 mask_path,
                 img_label,
                 tag]
            )


# ==============================================================================

# ==============================================================================
#                              ACTIVE LEARNING
# ==============================================================================


def al_split_glas(args):
    """
    Splits Glas dataset for active learning.
    It creates a validation/train sets in GlaS dataset.

    :param args:
    :return:
    """
    classes = ["benign", "malignant"]
    all_samples = []
    # Read the file Grade.csv
    baseurl = args.baseurl
    idcnt = 0.  # count the unique id for each sample
    with open(join(baseurl, "Grade.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # get rid of the header.
        for row in reader:
            # not sure why they thought it is a good idea to put a space
            # before the class. Now, I have to get rid of
            # it and possibly other hidden spaces ...
            row = [r.replace(" ", "") for r in row]
            msg = "The class `{}` is not within the predefined " \
                  "classes `{}`".format(row[2], classes)
            assert row[2] in classes, msg
            # name file, patient id, label
            all_samples.append([row[0], int(row[1]), row[2], idcnt])
            idcnt += 1.

    msg = "The number of samples {} do not match what they said " \
          "(165) .... [NOT OK]".format(len(all_samples))
    assert len(all_samples) == 165, msg

    # Take test samples aside. They are fix.
    test_samples = [[s[0], s[2], s[3]] for s in all_samples if s[0].startswith(
        "test")]
    msg = "The number of test samples {} is not 80 as they " \
          "said .... [NOT OK]".format(len(test_samples))
    assert len(test_samples) == 80, msg

    all_train_samples = [s for s in all_samples if s[0].startswith("train")]

    msg = "The number of train samples {} is not 85 " \
          "as they said .... [NOT OK]".format(len(all_train_samples))
    assert len(all_train_samples) == 85, msg

    patients_id = np.array([el[1] for el in all_train_samples])
    fig = plt.figure()
    plt.hist(patients_id, bins=np.unique(patients_id))
    plt.title("histogram-glas-train.")
    plt.xlabel("patient_id")
    plt.ylabel("number of samples")
    fig.savefig("tmp/glas-train.jpeg")
    # the number of samples per patient are highly unbalanced. so, we do not
    # split patients, but classes. --> we allow that samples from same
    # patient end up in train and valid. it is not that bad. it is just the
    # validation. plus, they are histology images. only the stain is more
    # likely to be relatively similar.

    all_train_samples = [[s[0], s[2], s[3]] for s in all_samples if s[
        0].startswith(
        "train")]

    benign = [s for s in all_train_samples if s[1] == "benign"]
    malignant = [s for s in all_train_samples if s[1] == "malignant"]

    # encode class name into int.
    dict_classes_names = {'benign': 0, 'malignant': 1}

    if not os.path.exists(args.fold_folder):
        os.makedirs(args.fold_folder)

    readme = "Format: float `id`: 0, str `img`: 1, None `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."
    # dump the readme
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    # Split
    splits = []
    for i in range(args.nbr_splits):
        for _ in range(1000):
            random.shuffle(benign)
            random.shuffle(malignant)
        splits.append({"benign": copy.deepcopy(benign),
                       "malignant": copy.deepcopy(malignant)}
                      )

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold
        contains a train, and valid set with a
        predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): w
        here each element is the list (str paths)
                 of the samples of each set: train, valid, and test,
                 respectively.
        """
        msg = "Something wrong with the provided sizes."
        assert len(lsamps) == s_tr + s_vl, msg

        # chunk the data into chunks of size ts
        # (the size of the test set), so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    def create_one_split(split_i, test_samples, benign, malignant, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param benign: list, list of train benign samples.
        :param malignant: list, list of train maligant samples.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        vl_size_benign = math.ceil(len(benign) * args.folding["vl"] / 100.)
        vl_size_malignant = math.ceil(
            len(malignant) * args.folding["vl"] / 100.)

        list_folds_benign = create_folds_of_one_class(
            benign, len(benign) - vl_size_benign, vl_size_benign)
        list_folds_malignant = create_folds_of_one_class(
            malignant, len(malignant) - vl_size_malignant, vl_size_malignant)

        msg = "We didn't obtain the same number of fold .... [NOT OK]"
        assert len(list_folds_benign) == len(list_folds_malignant), msg

        print("We found {} folds .... [OK]".format(len(list_folds_malignant)))

        outd = args.fold_folder
        for i in range(nbr_folds):
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv_glas(
                test_samples,
                join(out_fold, "test_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the train set
            train = list_folds_malignant[i][0] + list_folds_benign[i][0]
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv_glas(
                train,
                join(out_fold, "train_s_{}_f_{}.csv".format(split_i, i)),
                constants.U
            )

            # dump the valid set
            valid = list_folds_malignant[i][1] + list_folds_benign[i][1]
            dump_fold_into_csv_glas(
                valid,
                join(out_fold, "valid_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])
            # dump the coding.
            with open(join(out_fold, "encoding.yaml"), 'w') as f:
                yaml.dump(dict_classes_names, f)
            # dump the readme
            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    # Creates the splits
    for i in range(args.nbr_splits):
        create_one_split(
            i, test_samples, splits[i]["benign"], splits[i]["malignant"],
            args.nbr_folds)

    print("All GlaS splitting (`{}`) ended with success .... [OK]".format(
        args.nbr_splits))



def al_split_Caltech_UCSD_Birds_200_2011(args):
    """
    Create a validation/train sets in Caltech_UCSD_Birds_200_2011 dataset for
    active learning.
    Test set is provided.

    :param args:
    :return:
    """
    baseurl = args.baseurl
    classes_names, classes_id = [], []
    # Load the classes: id class
    with open(join(baseurl, "CUB_200_2011", "classes.txt"), "r") as fcl:
        content = fcl.readlines()
        for el in content:
            el = el.rstrip("\n\r")
            idcl, cl = el.split(" ")
            classes_id.append(idcl)
            classes_names.append(cl)
    # Load the images and their id.
    images_path, images_id = [], []
    with open(join(baseurl, "CUB_200_2011", "images.txt"), "r") as fim:
        content = fim.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, imgpath = el.split(" ")
            images_id.append(idim)
            images_path.append(imgpath)

    # Load the image labels.
    images_label = (np.zeros(len(images_path)) - 1).tolist()
    with open(join(baseurl, "CUB_200_2011", "image_class_labels.txt"),
              "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, clid = el.split(" ")
            # find the image index correspd. to the image id
            images_label[images_id.index(idim)] = classes_names[
                classes_id.index(clid)]

    # All what we need is in images_label, images_path.
    # classes_names will be used later to convert class name into integers.
    msg = "We expect Caltech_UCSD_Birds_200_2011 dataset to have " \
          "11788 samples. We found {} ... [NOT OK]".format(len(images_id))
    assert len(images_id) == 11788, msg
    all_samples = list(zip(images_path, images_label))  # Not used.

    # Split into train and test.
    all_train_samples = []
    test_samples = []
    idcnt = 0.  # count the unique id for each sample

    with open(join(baseurl, "CUB_200_2011", "train_test_split.txt"), "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, st = el.split(" ")
            img_idx = images_id.index(idim)
            img_path = images_path[img_idx]
            img_label = images_label[img_idx]
            filename, file_ext = os.path.splitext(img_path)
            mask_path = join("segmentations", filename + ".png")
            img_path = join("CUB_200_2011", "images", img_path)

            msg = "Image {} does not exist!".format(join(args.baseurl, img_path))
            assert os.path.isfile(join(args.baseurl, img_path)), msg

            msg = "Mask {} does not exist!".format(join(args.baseurl,
                                                        mask_path)
                                                   )
            assert os.path.isfile(join(args.baseurl, mask_path)), msg

            samplex = (img_path, mask_path, img_label, idcnt)
            if st == "1":  # train
                all_train_samples.append(samplex)
            elif st == "0":  # test
                test_samples.append(samplex)
            else:
                raise ValueError("Expected 0 or 1. "
                                 "Found {} .... [NOT OK]".format(st))

            idcnt += 1.

    print("Nbr. ALL train samples: {}".format(len(all_train_samples)))
    print("Nbr. test samples: {}".format(len(test_samples)))

    msg = "Something is wrong. We expected 11788. " \
          "Found: {}... [NOT OK]".format(
        len(all_train_samples) + len(test_samples))
    assert len(all_train_samples) + len(test_samples) == 11788, msg

    # Keep only the required classes:
    if args.nbr_classes is not None:
        fyaml = open(args.path_encoding, 'r')
        contyaml = yaml.load(fyaml)
        keys_l = list(contyaml.keys())
        indexer = np.array(list(range(len(keys_l)))).squeeze()
        select_idx = np.random.choice(indexer, args.nbr_classes, replace=False)
        selected_keys = []
        for idx in select_idx:
            selected_keys.append(keys_l[idx])

        # Drop samples outside the selected classes.
        tmp_all_train = []
        for el in all_train_samples:
            if el[2] in selected_keys:
                tmp_all_train.append(el)
        all_train_samples = tmp_all_train

        tmp_test = []
        for el in test_samples:
            if el[2] in selected_keys:
                tmp_test.append(el)

        test_samples = tmp_test

        classes_names = selected_keys

    # Train: Create dict where a key is the class name,
    # and the value is all the samples that have the same class.

    samples_per_class = dict()
    for cl in classes_names:
        samples_per_class[cl] = [el for el in all_train_samples if el[2] == cl]

    # Split
    splits = []
    print("Shuffling to create splits. May take some time...")
    for i in range(args.nbr_splits):
        for key in samples_per_class.keys():
            for _ in range(1000):
                random.shuffle(samples_per_class[key])
                random.shuffle(samples_per_class[key])
        splits.append(copy.deepcopy(samples_per_class))

    # encode class name into int.
    dict_classes_names = dict()
    for i in range(len(classes_names)):
        dict_classes_names[classes_names[i]] = i

    readme = "Format: float `id`: 0, str `img`: 1, None `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold
         contains a train, and valid set with a     predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set): where each
                 element is the list (str paths)
                 of the samples of each set: train, and valid, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the" \
                                           " provided sizes."

        # chunk the data into chunks of size ts (the size of the test set),
        # so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    def create_one_split(split_i, test_samples, c_split, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param c_split: dict, contains the current split.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        l_folds_per_class = []
        for key in c_split.keys():
            # count the number of tr, vl for this current class.
            vl_size = math.ceil(len(c_split[key]) * args.folding["vl"] / 100.)
            tr_size = len(c_split[key]) - vl_size
            # Create the folds.
            list_folds = create_folds_of_one_class(c_split[key], tr_size, vl_size)

            msg = "We did not get exactly {} folds, " \
                  "but `{}` .... [ NOT OK]".format(nbr_folds,  len(list_folds))
            assert len(list_folds) == nbr_folds, msg

            l_folds_per_class.append(list_folds)

        outd = args.fold_folder
        # Re-arrange the folds.
        for i in range(nbr_folds):
            print("\t Fold: {}".format(i))
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv_CUB(
                test_samples,
                join(out_fold, "test_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the train set
            train = []
            for el in l_folds_per_class:
                train += el[i][0]  # 0: tr
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv_CUB(
                train,
                join(out_fold, "train_s_{}_f_{}.csv".format(split_i, i)),
                constants.U
            )

            # dump the valid set
            valid = []
            for el in l_folds_per_class:
                valid += el[i][1]  # 1: vl

            dump_fold_into_csv_CUB(
                valid,
                join(out_fold, "valid_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])
            # dump the coding.
            with open(join(out_fold, "encoding.yaml"), 'w') as f:
                yaml.dump(dict_classes_names, f)

            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)
    # Creates the splits
    print("Starting splitting...")
    for i in range(args.nbr_splits):
        print("Split: {}".format(i))
        create_one_split(i, test_samples, splits[i], args.nbr_folds)

    print(
        "All Caltech_UCSD_Birds_200_2011 splitting (`{}`) ended with "
        "success .... [OK]".format(args.nbr_splits))


# ==============================================================================
#                              END: ACTIVE LEARNING
# ==============================================================================


def create_bin_mask_Oxford_flowers_102(args):
    """
    Create binary masks.
    :param args:
    :return:
    """
    def get_id(pathx, basex):
        """
        Get the id of a sample.
        :param pathx:
        :return:
        """
        rpath = relpath(pathx, basex)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        return id

    baseurl = args.baseurl
    imgs = find_files_pattern(join(baseurl, 'jpg'), '*.jpg')
    bin_fd = join(baseurl, 'segmim_bin')
    if not os.path.exists(bin_fd):
        os.makedirs(bin_fd)
    else:  # End.
        print('Conversion to binary mask has already been done. [OK]')
        return 0

    # Background color [  0   0 254]. (blue)
    print('Start converting the provided masks into binary masks ....')
    for im in tqdm.tqdm(imgs, ncols=80, total=len(imgs)):
        id_im = get_id(im, baseurl)
        mask = join(baseurl, 'segmim', 'segmim_{}.jpg'.format(id_im))
        assert isfile(mask), 'File {} does not exist. Inconsistent logic. .... [NOT OK]'.format(mask)
        msk_in = Image.open(mask, 'r').convert('RGB')
        arr_ = np.array(msk_in)
        arr_[:, :, 0] = 0
        arr_[:, :, 1] = 0
        arr_[:, :, 2] = 254
        blue = Image.fromarray(arr_.astype(np.uint8), mode='RGB')
        dif = ImageChops.subtract(msk_in, blue)
        x_arr = np.array(dif)
        x_arr = np.mean(x_arr, axis=2)
        x_arr = (x_arr != 0).astype(np.uint8)
        img_bin = Image.fromarray(x_arr * 255, mode='L')
        img_bin.save(join(bin_fd, 'segmim_{}.jpg'.format(id_im)), 'JPEG')


def al_split_Oxford_flowers_102(args):
    """
    Use the provided split:
    http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
    for active learning.

    :param args:
    :return:
    """
    baseurl = args.baseurl

    # splits
    fin = loadmat(join(baseurl, 'setid.mat'))
    trnid = fin['trnid'].reshape((-1)).astype(np.uint16)
    valid = fin['valid'].reshape((-1)).astype(np.uint16)
    tstid = fin['tstid'].reshape((-1)).astype(np.uint16)

    # labels
    flabels = loadmat(join(baseurl, 'imagelabels.mat'))['labels'].flatten()
    flabels -= 1  # labels are encoded from 1 to 102. We change that to
    # be from 0 to 101.

    # find all the files
    fdimg = join(baseurl, 'jpg')
    tr_set, vl_set, ts_set = [], [], []  # (img, mask, label (int))
    filesin = find_files_pattern(fdimg, '*.jpg')
    lid = []
    idcnt = 0.  # count the unique id for each sample
    for f in filesin:
        rpath = relpath(f, baseurl)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        mask = join(baseurl, 'segmim_bin', 'segmim_{}.jpg'.format(id))
        msg = 'File {} does not exist. Inconsistent logic. ' \
              '.... [NOT OK]'.format(mask)
        assert isfile(mask), msg
        rpath_mask = relpath(mask, baseurl)
        id = int(id)  # ids start from 1. Array indexing starts from 0.
        label = int(flabels[id - 1])
        sample = (rpath, rpath_mask, label, idcnt)
        lid.append(id)
        if id in trnid:
            tr_set.append(sample)
        elif id in valid:
            vl_set.append(sample)
        elif id in tstid:
            ts_set.append(sample)
        else:
            raise ValueError('ID:{} not found in train, valid, test. '
                             'Inconsistent logic. ....[NOT OK]'.format(id))

        idcnt += 1.

    dict_classes_names = dict()
    uniquel = np.unique(flabels)
    for i in range(uniquel.size):
        dict_classes_names[str(uniquel[i])] = int(uniquel[i])

    outd = args.fold_folder
    out_fold = join(outd, "split_{}/fold_{}".format(0, 0))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    readme = "Format: float `id`: 0, str `img`: 1, None `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."

    # shuffle train
    for t in range(1000):
        random.shuffle(tr_set)

    dump_fold_into_csv_OXF(tr_set,
                           join(out_fold, "train_s_{}_f_{}.csv".format(0, 0)),
                           constants.U
                           )
    dump_fold_into_csv_OXF(vl_set,
                           join(out_fold, "valid_s_{}_f_{}.csv".format(0, 0)),
                           constants.L
                           )
    dump_fold_into_csv_OXF(ts_set,
                           join(out_fold, "test_s_{}_f_{}.csv".format(0, 0)),
                           constants.L
                           )

    # current fold.

    # dump the coding.
    with open(join(out_fold, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    # dump the seed
    with open(join(out_fold, "seed.txt"), 'w') as fx:
        fx.write("MYSEED: " + os.environ["MYSEED"])

    with open(join(out_fold, "readme.md"), 'w') as fx:
        fx.write(readme)

    # folder of folds

    # readme
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    print(
        "Oxford_flowers_102 splitting (`{}`) ended with "
        "success .... [OK]".format(0))


def al_split_camelyon16(args):
    """
    Use the provided split:
    https://github.com/jeromerony/survey_wsl_histology/blob/master/
    datasets-split/README.md
    for active learning.

    :param args:
    :return:
    """

    def csv_loader(fname):
        """
        Read a *.csv file. Each line contains:
         1. img: str
         2. mask: str or '' or None
         3. label: str

        :param fname: Path to the *.csv file.
        :param rootpath: The root path to the folders of the images.
        :return: List of elements.
        Each element is the path to an image: image path, mask path [optional],
        class name.
        """
        with open(fname, 'r') as f:
            out = [
                [row[0],
                 row[1] if row[1] else None,
                 row[2]
                 ]
                for row in csv.reader(f)
            ]

        return out

    csv_df = 'folds/camelyon16-split-0-fold-0-512-512-survey'
    # load survey csv files.
    trainset = csv_loader(join(csv_df, 'train_s_0_f_0.csv'))
    validset = csv_loader(join(csv_df, 'valid_s_0_f_0.csv'))
    testset = csv_loader(join(csv_df, 'test_s_0_f_0.csv'))

    baseurl = args.baseurl

    # find all the files
    fdimg = join(baseurl, 'jpg')
    tr_set, vl_set, ts_set = [], [], []
    idcnt = 0.  # count the unique id for each sample

    stats = {
        'train': {
            'normal': 0.,
            'tumor': 0.
        },
        'valid': {
            'normal': 0.,
            'tumor': 0.
        },
        'test': {
            'normal': 0.,
            'tumor': 0.
        }
    }

    # train
    for f in trainset:
        img = f[0]
        mask = f[1]
        label = f[2]
        tr_set.append((idcnt, img, mask, label))
        idcnt += 1.
        if label == 'normal':
            stats['train']['normal'] += 1.
        else:
            stats['train']['tumor'] += 1.

    # valid
    for f in validset:
        img = f[0]
        mask = f[1]
        label = f[2]
        vl_set.append((idcnt, img, mask, label))
        idcnt += 1.

        if label == 'normal':
            stats['valid']['normal'] += 1.
        else:
            stats['valid']['tumor'] += 1.

    # test
    for f in testset:
        img = f[0]
        mask = f[1]
        label = f[2]
        ts_set.append((idcnt, img, mask, label))
        idcnt += 1.

        if label == 'normal':
            stats['test']['normal'] += 1.
        else:
            stats['test']['tumor'] += 1.

    dict_classes_names = {"normal": 0, "tumor": 1}

    outd = args.fold_folder
    out_fold = join(outd, "split_{}/fold_{}".format(0, 0))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    readme = "Format: float `id`: 0, str `img`: 1, None `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."

    # shuffle train
    for t in range(1000):
        random.shuffle(tr_set)

    dump_fold_into_csv_CAM16(tr_set,
                             join(out_fold, "train_s_{}_f_{}.csv".format(0, 0)),
                             constants.U
                             )
    dump_fold_into_csv_CAM16(vl_set,
                             join(out_fold, "valid_s_{}_f_{}.csv".format(0, 0)),
                             constants.L
                             )
    dump_fold_into_csv_CAM16(ts_set,
                             join(out_fold, "test_s_{}_f_{}.csv".format(0, 0)),
                             constants.L
                             )

    # current fold.

    # dump the coding.
    with open(join(out_fold, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    # dump the seed
    with open(join(out_fold, "seed.txt"), 'w') as fx:
        fx.write("MYSEED: " + os.environ["MYSEED"])

    with open(join(out_fold, "readme.md"), 'w') as fx:
        fx.write(readme)

    with open(join(out_fold, "stats-sets.yaml"), 'w') as fx:
        total = sum([stats[el]['normal'] + stats[el]['tumor'] for el in
                   list(stats.keys())])
        stats['total_samples'] = total
        yaml.dump(stats, fx)
        print("Stats:", stats)

    # folder of folds

    # readme
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    print("camelyon16 splitting (`{}`) ended with success .... [OK]".format(0))

# ==============================================================================
#                               RUN
# ==============================================================================


def do_glas():
    """
    GlaS.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============
    reproducibility.init_seed()

    announce_msg("Processing dataset: {}".format(constants.GLAS))

    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': constants.GLAS})),
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "glas",
            "fold_folder": "folds/glas",
            "img_extension": "bmp",
            # nbr_splits: how many times to perform the k-folds over
            # the available train samples.
            "nbr_splits": 1
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])

    reproducibility.init_seed()
    al_split_glas(Dict2Obj(args))
    get_stats(Dict2Obj(args), split=0, fold=0, subset='train')


def do_Caltech_UCSD_Birds_200_2011():
    """
    Caltech-UCSD-Birds-200-2011.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.init_seed()

    # ===========================

    announce_msg("Processing dataset: {}".format(constants.CUB))

    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': constants.CUB})),
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "Caltech-UCSD-Birds-200-2011",
            "fold_folder": "folds/Caltech-UCSD-Birds-200-2011",
            "img_extension": "bmp",
            "nbr_splits": 1,  # how many times to perform the k-folds over
            # the available train samples.
            "path_encoding": "folds/Caltech-UCSD-Birds-200-2011/encoding-origine.yaml",
            "nbr_classes": None  # Keep only 5 random classes. If you want
            # to use the entire dataset, set this to None.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])
    reproducibility.init_seed()
    al_split_Caltech_UCSD_Birds_200_2011(Dict2Obj(args))
    get_stats(Dict2Obj(args), split=0, fold=0, subset='train')


def do_Oxford_flowers_102():
    """
    Oxford-flowers-102.
    The train/valid/test sets are already provided.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.init_seed()

    # ===========================

    announce_msg("Processing dataset: {}".format(constants.OXF))
    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': constants.OXF})),
            "dataset": "Oxford-flowers-102",
            "fold_folder": "folds/Oxford-flowers-102",
            "img_extension": "jpg",
            "path_encoding": "folds/Oxford-flowers-102/encoding-origine.yaml"
            }
    # Convert masks into binary masks: already done.
    # create_bin_mask_Oxford_flowers_102(Dict2Obj(args))
    reproducibility.init_seed()
    al_split_Oxford_flowers_102(Dict2Obj(args))
    get_stats(Dict2Obj(args), split=0, fold=0, subset='train')


def do_camelyon16():
    """
    camelyon16.
    The train/valid/test sets are already provided.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.init_seed()

    # ===========================

    ds = constants.CAM16
    announce_msg("Processing dataset: {}".format(ds))
    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': ds})),
            "dataset": ds,
            "fold_folder": "folds/{}".format(ds),
            "img_extension": "jpg",
            "path_encoding": "folds/{}/encoding-origine.yaml".format(ds)
            }
    # Convert masks into binary masks: already done.
    # create_bin_mask_Oxford_flowers_102(Dict2Obj(args))
    reproducibility.init_seed()
    al_split_camelyon16(Dict2Obj(args))
    # get_stats(Dict2Obj(args), split=0, fold=0, subset='train')




if __name__ == "__main__":
    check_if_allow_multgpu_mode()

    # ==========================================================================
    #                          ACTIVE LEARNING
    # ==========================================================================
    do_glas()
    do_Caltech_UCSD_Birds_200_2011()
    # do_Oxford_flowers_102()
    # do_camelyon16()
    # ==========================================================================
    #                       END: ACTIVE LEARNING
    # ==========================================================================


