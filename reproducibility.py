# self-contained-as-possible module.
# handles reproducibility procedures.

import random
import os
import warnings


import numpy as np
import torch
from torch._C import default_generator


DEFAULT_SEED = 0   # the default seed.


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


def get_seed():
    """
    Get the default seed from the environment variable.
    If not set, we use our default seed.
    :return: int, a seed.
    """
    try:
        msg = "REQUIRED SEED: {}  ".format(os.environ["MYSEED"])
        announce_msg(msg)

        return int(os.environ["MYSEED"])
    except KeyError:
        print("`os.environ` does not have a key named `MYSEED`."
              "This key is supposed to hold the current seed. Please set it,"
              "and try again, if you want.")

        warnings.warn("MEANWHILE, .... WE ARE GOING TO USE OUR DEFAULT SEED: "
                      "{}".format(DEFAULT_SEED))
        os.environ["MYSEED"] = str(DEFAULT_SEED)
        msg = "DEFAULT SEED: {}  ".format(os.environ["MYSEED"])
        announce_msg(msg)
        return DEFAULT_SEED


def init_seed(seed=None):
    """
    * initialize the seed.
    * Set a seed to some modules for reproducibility.

    Note:

    While this attempts to ensure reproducibility, it does not offer an
    absolute guarantee. The results may be similar to some precision.
    Also, they may be different due to an amplification to extremely
    small differences.

    See:

    https://pytorch.org/docs/stable/notes/randomness.html
    https://stackoverflow.com/questions/50744565/
    how-to-handle-non-determinism-when-training-on-a-gpu

    :param seed: int, a seed. Default is None: use the default seed (0).
    :return:
    """
    if seed is None:
        seed = get_seed()
    else:
        os.environ["MYSEED"] = str(seed)
        announce_msg("SEED: {} ".format(os.environ["MYSEED"]))

    check_if_allow_multgpu_mode()
    reset_seed(seed)


def reset_seed(seed, check_cudnn=True):
    """
    Reset seed to some modules.
    :param seed: int. The current seed.
    :param check_cudnn: boo. if true, we check if we are in multi-gpu to
    disable the cudnn use. `ALLOW_MULTIGPUS` variable has to be already
    created in os.environ otherwise an error will be raise.
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Deterministic mode can have a performance impact, depending on your
    torch.backends.cudnn.deterministic = True
    # model: https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    # If multigpu is on, deactivate cudnn since it has many random things
    # that we can not control.
    if check_cudnn:
        cond = torch.cuda.device_count() > 1
        cond = cond and (os.environ["ALLOW_MULTIGPUS"] == 'True')
        if cond:
            torch.backends.cudnn.enabled = False


def set_default_seed():
    """
    Set the default seed.
    :return:
    """
    assert "MYSEED" in os.environ.keys(), "`MYSEED` key is not found in " \
                                          "os.environ.keys() ...." \
                                          "[NOT OK]"
    reset_seed(int(os.environ["MYSEED"]))


def manual_seed(seed):
    r"""Sets the seed for generating random numbers. Returns a
    `torch._C.Generator` object.

    NOTE: WE REMOVE MANUAL RESEEDING ALL THE GPUS. At this point,
    it is not necessary; and there is not logic/reason
    to do it since we want only to reseed the current device.

    Args:
        seed (int): The desired seed.
    """
    return default_generator.manual_seed(int(seed))


def force_seed_thread(seed):
    """
    For seed to some modules.
    :param seed:
    :return:
    """
    manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # Deterministic mode can have a performance impact, depending on your
    # torch.backends.cudnn.deterministic = True
    # model: https://pytorch.org/docs/stable/notes/randomness.html#cudnn
