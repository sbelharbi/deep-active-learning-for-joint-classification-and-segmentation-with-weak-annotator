# This module shouldn't import any of our modules to avoid recursive importing.
import os
import argparse
import textwrap
import csv
from os.path import join
import fnmatch


from sklearn.metrics import auc
import torch
import numpy as np


CONST1 = 1000  # used to generate random numbers.


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


def announce_msg(msg, upper=True):
    """
    Display sa message in the standard output. Something like this:
    =================================================================
                                message
    =================================================================

    :param msg: str, text message to display.
    :param upper: True/False, if True, the entire message is converted into
    uppercase. Else, the message is displayed
    as it is.
    :return: str, what was printed in the standard output.
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])

    print(output_msg)

    return output_msg


def check_if_allow_multgpu_mode():
    """
    Check if we can do multigpu.
    If yes, allow multigpu.
    :return: ALLOW_MULTIGPUS: bool. If True, we enter multigpu mode:
    1. Computation will be dispatched over the AVAILABLE GPUs.
    2. Synch-BN is activated.
    """
    if "CC_CLUSTER" in os.environ.keys():
        ALLOW_MULTIGPUS = True  # CC.
    else:
        ALLOW_MULTIGPUS = False  # others.

    # ALLOW_MULTIGPUS = True
    os.environ["ALLOW_MULTIGPUS"] = str(ALLOW_MULTIGPUS)
    NBRGPUS = torch.cuda.device_count()
    ALLOW_MULTIGPUS = ALLOW_MULTIGPUS and (NBRGPUS > 1)

    return ALLOW_MULTIGPUS


def check_tensor_inf_nan(tn):
    """
    Check if a tensor has any inf or nan.
    """
    if any(torch.isinf(tn.view(-1))):
        raise ValueError("Found inf in projection.")
    if any(torch.isnan(tn.view(-1))):
        raise ValueError("Found nan in projection.")


def wrap_command_line(cmd):
    """
    Wrap command line
    :param cmd: str. command line with space as a separator.
    :return:
    """
    return " \\\n".join(textwrap.wrap(
        cmd, width=77, break_long_words=False, break_on_hyphens=False))


def drop_normal_samples(l_samples):
    """
    Remove normal samples from the list of samples.

    When to call this?
    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.

    :param l_samples: list of samples resulting from csv_loader().
    :return: l_samples without any normal sample.
    """
    return [el for el in l_samples if el[3] == 'tumor']


def csv_loader(fname, rootpath, drop_normal=False):
    """
    Read a *.csv file. Each line contains:
     0. id_: str
     1. img: str
     2. mask: str or '' or None
     3. label: str
     4. tag: int in {0, 1}

     Example: 50162.0, test/img_50162_label_frog.jpeg, , frog, 0

    :param fname: Path to the *.csv file.
    :param rootpath: The root path to the folders of the images.
    :return: List of elements.
    :param drop_normal: bool. if true, normal samples are dropped.
    Each element is the path to an image: image path, mask path [optional],
    class name.
    """
    with open(fname, 'r') as f:
        out = [
            [row[0],
             join(rootpath, row[1]),
             join(rootpath, row[2]) if row[2] else None,
             row[3],
             int(row[4])
             ]
            for row in csv.reader(f)
        ]

    if drop_normal:
        out = drop_normal_samples(out)

    return out


def csv_writer(data, fname):
    """
    Write a list of rows into a file.
    """
    msg = "'data' must be a list. found {}".format(type(data))
    assert isinstance(data, list), msg

    with open(fname, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            filewriter.writerow(row)


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


def check_nans(tens, msg=''):
    """
    Check if the tensor 'tens' contains any 'nan' values, and how many.

    :param tens: torch tensor.
    :param msg: str. message to display if there is nan.
    :return:
    """
    nbr_nans = torch.isnan(tens).float().sum().item()
    if nbr_nans > 0:
        print("NAN-CHECK: {}. Found: {} NANs.".format(msg, nbr_nans))


def compute_auc(vec, nbr_p):
    """
    Compute the area under a curve.
    :param vec: vector contains values in [0, 100.].
    :param nbr_p: int. number of points in the x-axis. it is expected to be
    the same as the number of values in `vec`.
    :return: float in [0, 100]. percentage of the area from the perfect area.
    """
    if vec.size == 1:
        return float(vec[0])
    else:
        area_under_c = auc(x=np.array(list(range(vec.size))), y=vec)
        area_under_c /= (100. * (nbr_p - 1))
        area_under_c *= 100.  # (%)
        return area_under_c

# ==============================================================================
#                                            TEST
# ==============================================================================


def test_announce_msg():
    """
    Test announce_msg()
    :return:
    """
    announce_msg("Hello world!!!")