import os
from os.path import join
import collections
import copy
import warnings
import datetime as dt

import PIL
from PIL import Image
import numpy as np
import tqdm
import pickle as pkl
from PIL import ImageEnhance


import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


import reproducibility
import constants


__all__ = ["PhotoDataset", "_init_fn", "default_collate", "MyDataParallel"]


def _init_fn(worker_id):
    """
    Init. function for the worker in dataloader.
    :param worker_id:
    :return:
    """
    pass  # no longer necessary since we override the process seed.


def default_collate(batch):
    """
    Override
    https://pytorch.org/docs/stable/_modules/torch/utils/data/
    dataloader.html#DataLoader

    Reference:
    def default_collate(batch) at
    https://pytorch.org/docs/stable/_modules/torch/utils/data/
    dataloader.html#DataLoader
    https://discuss.pytorch.org/t/
    how-to-create-a-dataloader-with-variable-size-input/8278/3
    https://github.com/pytorch/pytorch/issues/1512

    We need our own collate function that wraps things up
    (id_, img, mask, label, tag, soft_labels_trg).

    In this setup,  batch is a list of tuples (the result of calling:
    img, mask, label = PhotoDataset[i].
    The output of this function is four elements:
        * data: a pytorch tensor of size (batch_size, c, h, w) of float32 .
        Each sample is a tensor of shape (c, h_, w_) that represents a cropped
        patch from an image (or the entire image) where: c is the depth of the
        patches (since they are RGB, so c=3),  h is the height of the patch,
        and w_ is the its width.
        * mask[optional=None]: a list of pytorch tensors of size (batch_size,
        1, h, w) full of 1 and 0. The mask of the ENTIRE image (no cropping is
        performed). Images does not have the same size, and the same thing
        goes for the masks. Therefore, we can't put the masks in one tensor.
        If there is no mask in the ground truth, return None.
        * target: a vector (pytorch tensor) of length batch_size of type
        torch.LongTensor containing the image-level labels.
    :param batch: list of tuples (img, mask [optional=None], label)
    :return: 3 elements: tensor data, list of tensors of masks, tensor of labels.
    """
    # 0-ID
    # ids = torch.from_numpy(np.array([item[0] for item in batch],
    #                                 dtype=np.float32))
    ids = [item[0] for item in batch]
    # 1-Image
    imgs = torch.stack([item[1] for item in batch])
    # 2-Mask
    if batch[0][2] is not None:
        # masks = [item[2] for item in batch]  # each element is of size (1, h,
        # w).
        masks = torch.stack([item[2] for item in batch])
    else:
        masks = None
    # 3-Label
    label = torch.LongTensor([item[3] for item in batch])
    # 4-Tag
    tag = torch.LongTensor([item[4] for item in batch])

    # 5- cropping coordinates: list of tuple (i, j, h, w) or None
    crop_cord = [item[5] for item in batch]

    return ids, imgs, masks, label, tag, crop_cord


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *inputs, **kwargs):
        """
        The exact same as in Pytorch.
        We use it for debugging.
        :param inputs:
        :param kwargs:
        :return:
        """
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        tx = dt.datetime.now()
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # print("Scattering took {}".format(dt.datetime.now() - tx))
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        tx = dt.datetime.now()
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        # print("Replicating took {}".format(dt.datetime.now() - tx))
        tx = dt.datetime.now()
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        # print("Gathering took {}".format(dt.datetime.now() - tx))
        return self.gather(outputs, self.output_device)


class MyRandomCropper(transforms.RandomCrop):
    """
    Crop the given PIL Image at a random location.

    Class inherits from transforms.RandomCrop().
    It does exactly the same thing, except, it returns the coordinates of
    along with the crop.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
            Coordinates of the crop: tuple (i, j, h, w).
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return TF.crop(img, i, j, h, w), (i, j, h, w)


class PhotoDataset(Dataset):
    """
    Class that overrides torch.utils.data.Dataset.
    """
    def __init__(self,
                 data,
                 dataset_name,
                 name_classes,
                 transform_tensor,
                 set_for_eval,
                 transform_img=None,
                 resize=None,
                 resize_h_to=None,
                 resize_mask=False,
                 crop_size=None,
                 padding_size=None,
                 padding_mode="reflect",
                 up_scale_small_dim_to=None,
                 do_not_save_samples=False,
                 ratio_scale_patch=1.,
                 for_eval_flag=False,
                 scale_algo=Image.LANCZOS,
                 enhance_color=False,
                 enhance_color_fact=1.,
                 check_ps_msk_path=False,
                 previous_pairs=None,
                 fd_p_msks=None
                 ):
        """
        :param data: A list of str absolute paths of the images of dataset.
               In this case, no preprocessing will be used (such brightness
               standardization, ...). Raw data will be used
               directly.
        :param dataset_name: str, name of the dataset.
        :param name_classes: dict, {"classe_name": int}.
        :param transform_tensor: a composition of transforms that performs over
               torch.tensor: torchvision.transforms.Compose(). or None.
        :param set_for_eval: True/False. Used to entirely prepare the data for
               evaluation by performing all the necessary steps to get the
               data ready for input to the model. Useful for the evaluation
               datasets such as the validation set or the test test. If True we
               do all that, else the preparation of the data is done when
               needed. If  dataset is LARGE AND you increase the size of the
               samples through a processing step (upscaling for instance), we
               recommend to set this to False since you will need a large
               memory. In this case, the validation will be slow (you can
               increase the number of workers if you use a batch size of 1).
        :param transform_img: a composition of transforms that performs over
               images: torchvision.transforms.Compose(). or None.
        :param resize: int, or sequence of two int (w, h), or None. The size
               to which the original image is resized. If None, the original
               image is used. (needed only when data is a list). exclusive with
               'resize_h_to'. operates only on the image and not the mask
                unless `resize_mask` is explicitly set to true.
        :param resize_h_to: int or None. resize the original image height into
               this value. the width will be computed accordingly to preserve
               the proportions. this is an alternative to 'resize'. they can
               not be both set. they are exclusive. this resize is done on
               the image before doing anything else. operates only on the
               image and not the mask unless `resize_mask` is explicitly set
               to true.
        :param resize_mask: bool. if true, the original mask is resized to
        'resize' or `resize_h_to` before doing anything else. useful only for
        particular cases.
        :param crop_size: int or tuple of int (h, w) or None. Size of the
               cropped patches. If int, the size will be the same for height
               and width. If None, no cropping is done, and the entire image
               is used (such the case in validation).
        :param padding_size: (h%, w%), how much to pad (top/bottom) and
               (left/right) of the ORIGINAL IMAGE. h, w are percentages or
               None. you can not pad only one side.
        :param padding_mode: str, accepted padding mode (
               https://pytorch.org/docs/stable/torchvision/transforms.html
               #torchvision.transforms.functional.pad)
        :param up_scale_small_dim_to: int or None. If not None, we upscale the
               small dimension (height or width) to this value (then compute
               the upscale ration r). Then, upscale the other dimension to a
               proportional value (using the ratio r). This is helpful when the
               images have small size such as in the dataset
               Caltech-UCSD-Birds-200-2011. Due to the depth of the model,
               small images may 'disappear' or provide a very small
               attention map. To upscale, use the algorithm in `scale_algo`.
        :param scale_algo: int, resize algorithm. Possible choices:
               Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING,
              Image.BICUBIC, Image.LANCZOS. Default: Image.LANCZOS. See:
              https://pillow.readthedocs.io/en/stable/reference/
              Image.html#PIL.Image.Image.resize
        :param do_not_save_samples: Bool. If True, we do not save samples in
               memory. The default behavior of the code is to preload the
               samples, and save them in memory to speedup access and to avoid
               disc access. However, this may be impractical when dealing with
               large dataset during the final processing (evaluation). It is
               not necessary to keep the samples of the dataset in the memory
               once they are processed. Consequences to this boolean flag: If
               it is True, we do not preload sample (read from disc), AND once a
               sample is loaded, it is not stored. There few things that we
               save: 1. The size of the sample (h, w). in
               self.original_images_size. We remind that this flag is useful
               only at the end of the training when you want to evaluate on a
               set (train, valid, test). In this case, there is no need to store
               anything. If the dataset is large, this will cause memory
               overflow (in case you run your code on a server with reserved
               amount of memory). If you set this flag to True, use 0 workers
               for the dataloader, since we will be processing the samples
               sequentially, and we want to avoid to load a sample ahead
               (no point of doing that).
               This has another usefullness when dealing with large train
               dataset and using multiprocessing. You will need a huge memory to
               work. To keep using myltiprocessing we recommend setting this
               to True so you do not save the image in memory (
               multiprocessing duplicates the main process memory), but you
               read from disc all the time, but you need extremely fast disc
               access. It a compromise either you have an insane size of RAM
               per job or you have a fast disc access.
        :param ratio_scale_patch: the ratio to which the cropped patch is
               scaled. during evaluation, original images are also rescaled
               using this ratio. if you want to keep the cropped patch as it
               is, set this variable to 1.
        :param for_eval_flag: bool. Set this to true of the dataset is
               intended for evaluation. This is different from
               `set_for_eval`. The latter affects how we process the images,
               while this variable just tells us if this dataset will be used
               evaluation or not.
        :param enhance_color: bool. if true, the color of an image is enhanced.
        :param enhance_color_fact: float [0., [. color enhancing factor. 0.
               gives black and white. 1 gives the same image. low values than 1
               means less color, higher values than 1 means more colors.
        :param check_ps_msk_path: bool. default: False. if true, the ground
               truth masks are retrieved from `fd_p_msks` instead of the
               original folder. this is helpful when using an alternative
               ground truth.
        :param previous_pairs: dict. contains keys (key: id_u, val: id_l)
               pairs of unlabeled samples that have been pseudo-labeled.
               in the unlabeled case, the name of the pseudo-mask is
               'fd_p_msks/{}-{}.bmp'.format(id_u, id_l).
               `check_ps_msk_path` must be true.
               See `self.get_original_input_mask` on the internal rule of
               getting the segmentation mask.
        :param fd_p_msks: str absolute path. default: None. folder where to
               look for the alternative ground truth masks.
        lool for
        """
        msg = "'scale_algo' must be an int. found {}.".format(type(scale_algo))
        assert isinstance(scale_algo, int), msg
        algos = [Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING,
                 Image.BICUBIC, Image.LANCZOS]
        msg = "'scale_algo' must be in {}. found {}.".format(scale_algo, algos)
        assert scale_algo in algos, msg

        assert isinstance(
            data, list), "`data` is supposed to be of type: list. " \
                         "Found {}".format(type(data))

        msg = "must 0 < `ration_scale_patch` <=1. found {} ...[NOT " \
              "OK]".format(ratio_scale_patch)
        assert 0. < ratio_scale_patch <= 1., msg

        msg = "'name_classes' must be a dict. Found {}".format(
            type(name_classes))
        assert isinstance(name_classes, dict), msg

        msg = "'resize_h_to' and 'resize' are exclusive. resize_h_to={}, " \
              "'resize'={}.".format(resize_h_to, resize)
        assert not all([resize_h_to is not None, resize is not None]), msg

        if resize_h_to is not None:
            msg = "'resize_h_to' must be of type int. found {}.".format(
                type(resize_h_to)
            )
            assert isinstance(resize_h_to, int), msg
            msg = "resize_h_to must be > 0. found {}.".format(resize_h_to)
            assert resize_h_to > 0, msg

        if resize_mask:
            msg = "'resize_mask' is true while non of 'resize', " \
                  "'resize_h_to' is set. one has to be set."
            assert any([resize_h_to is not None, resize is not None]), msg

        msg = "'enhance_color_fact' must be >= 0. found {}.".format(
            enhance_color_fact)
        assert enhance_color_fact >= 0., msg

        self.check_ps_msk_path = check_ps_msk_path
        self.previous_pairs = previous_pairs
        self.fd_p_msks = fd_p_msks

        self.resize_h_to = resize_h_to
        self.resize_mask = resize_mask
        self.ratio_scale_patch = ratio_scale_patch
        self.enhance_color = enhance_color
        self.enhance_color_fact = enhance_color_fact
        # convert mask to tensor.
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        self.set_for_eval = set_for_eval
        self.set_for_eval_backup = set_for_eval
        self.for_eval_flag = for_eval_flag
        self.name_classes = name_classes
        self.up_scale_small_dim_to = up_scale_small_dim_to
        self.do_not_save_samples = do_not_save_samples
        self.scale_algo = scale_algo

        allowed_datasets = ["bach-part-a-2018",
                            "fgnet",
                            "afad-lite",
                            "afad-full",
                            "historical-color-image-decade",
                            constants.SVHN,
                            constants.CIFAR_10,
                            constants.CIFAR_100,
                            constants.MNIST,
                            constants.GLAS,
                            constants.OXF,
                            constants.CUB,
                            constants.CAM16
                            ]
        msg = "dataset_name = {} unsupported. Please double check. We do " \
              "some operations that are dataset dependent. So, you may " \
              "need to do these operations on your own (mask binarization, " \
              "...). Exiting .... [NOT OK]".format(dataset_name)
        assert dataset_name in allowed_datasets, msg

        # only medical datasets are allowed to be padded since it makes sens.
        # if dataset_name != "bach-part-a-2018":
        #     assert padding_size is None, "We do not support padding " \
        #                                  "train/test for data other than " \
        #                                  "`bach-part-a-2018` set."

        self.dataset_name = dataset_name
        self.samples = data

        self.indx_of_id = dict()  # used to fetch a sample based on their id.
        for i, sam in enumerate(data):
            self.indx_of_id[sam[0]] = i

        self.seeds = None
        self.set_up_new_seeds()  # set up seeds for the initialization.

        self.transform_img = transform_img
        self.transform_tensor = transform_tensor
        self.resize = None
        if resize:
            if isinstance(resize, int):
                self.resize = (resize, resize)
            elif isinstance(self.resize, collections.Sequence):
                self.resize = resize

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        if crop_size:
            self.randomCropper = MyRandomCropper(size=crop_size, padding=0)
        else:
            self.randomCropper = None

        # padding is valid only if both are not None.
        # we do not deal with padding only one side. sorry.
        if (padding_size[0] is None and padding_size[1] is not None) or (
                padding_size[1] is None and padding_size[0] is not None):
            padding_size = None
            warnings.warn("You asked to pad only one side. Sorry,"
                          "we do not support this config. we canceled the "
                          "padding. you have to pad both sides or non.")

        if (padding_size[0] is None) and (padding_size[1] is None):
            padding_size = None

        self.padding_size = padding_size
        if padding_size:
            msg = "You asked for padding, but you didn't specify the " \
                  "padding mode. Accepted modes can be found in " \
                  "https://pytorch.org/docs/stable/torchvision/transforms." \
                  "html#torchvision.transforms.functional.pad"
            assert padding_mode is not None, msg
        self.padding_mode = padding_mode

        # self.n = len(self.samples)

        self.images = []
        self.original_images_size = [None for _ in range(len(self))]
        self.absolute_paths_imgs = []
        self.absolute_paths_masks = []

        # Expected format of each line in samples:
        # 1. id: float, a unique id of the sample within the entire dataset.
        # 2. path_img: str, path to the image.
        # 3. path_mask: str or None, path to the mask if there is any.
        # Otherwise, None.
        # 4. label: int, the class label of the sample.
        # 5. tag: int in {0, 1, 2} where: 0: the samples belongs to the
        # supervised set (L). 1: The  sample belongs to the unsupervised set
        # (U). 2: The sample belongs to the set of newly labeled samples (
        # L'). This sample came from U and was labeling following a specific
        # mechanism. In this class, samples can have only one of these tags:
        # {0, 2}.

        # position of each element. can be helpful to access to each one
        # across the class.
        self.position = {
            "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4, "crop_cord": 5}
        self.possible_tags = constants.samples_tags  # [0, 1, 2]  # where
        # 0: labeled
        # 1: unlabeled
        # 2: labeled but came from unlabeled set.

        msg = "Your data does not follow the expected format. Each row must " \
              "has 5 elements: id, path_img, path_mask, label, tag." \
              " We found {} ....[NOT OK]".format(len(self.samples[0]))
        assert len(self.samples[0]) == 5, msg

        nbr_classes = len(list(self.name_classes.keys()))

        tik = 0
        self.list_ids = []  # need them to be ready.
        for idxx, path_img, path_mask, _, _ in self.samples:
            self.absolute_paths_imgs.append(path_img)
            self.absolute_paths_masks.append(path_mask)
            self.list_ids.append(idxx)

        self.labels = []
        self.masks = []
        self.tags = []  # holds the sample tag.
        self.ids = []  # holds ids.
        self.preloaded = False

        if not do_not_save_samples:
            self.preload_images()

        self.inputs_ready = []
        self.labels_ready = []
        self.masks_ready = []
        self.tags_ready = []
        self.ids_ready = []

        if self.set_for_eval:
            self.set_ready_eval_fn()

    @property
    def n(self):
        """
        Number of samples.
        :return: int, the total number of samples in this dataset.
        """
        return len(self.samples)

    def add_u_to_lp_from_set(self, ids_to_move, source):
        """
        Import samples identified by their ids from another dataset (U) into
        this dataset (L`).

        The updates are done directly over this dataset only. The other
        datasets will be updated using self.change_u_to_lp().

        Note: this function changes the rng status of numpy. It is has to be
        locked down between re-seeders.

        :param ids_to_move: list contains samples ids to be moved.
        :param source: instance of this class.
        :return: nothing.
        """
        msg = "`ids_to_move` is empty."
        assert ids_to_move, msg

        nbr = 0

        for i in range(len(source.samples)):
            sample = copy.deepcopy(source.samples[i])
            idx = sample[source.position['id']]

            if idx in ids_to_move:
                tag = sample[self.position['tag']]
                msg = "We expect the sample with the id {} to be unlabeled " \
                      "(tag = 1), but found tag = {}.".format(idx, tag)
                assert tag == constants.U, msg

                sample[self.position["tag"]] = constants.PL  # change its tag
                # to L'.
                self.samples.append(sample)  # then move it. This kind of
                # moving is still bad because sample in L' are stacked after
                # each other. Later, the sampling from dataset will be
                # biased. A better way it to mix the samples L, and L'. But
                # this requires changing many-many things. TODO some-day.

                # add the corresponding seed
                self.seeds.append(np.random.randint(0, 100000, 1)[0])

                if self.use_soft_trg:
                    self.soft_labels_trg.append(None)
                else:
                    path_soft_trg = join(self.fd_soft_trg,
                                         '{}.pkl'.format(idx))
                    self.soft_labels_trg.append(path_soft_trg)

                self.original_images_size.append(None)
                self.absolute_paths_imgs.append(sample[self.position['img']])
                self.absolute_paths_masks.append(sample[self.position['mask']])

                if not self.do_not_save_samples:
                    stufxx = self.load_sample_i(-1)
                    id_, img, mask, label, tag, soft_labels_trg = stufxx

                    self.images.append(img)
                    self.masks.append(mask)
                    self.labels.append(label)
                    self.ids.append(id_)
                    self.tags.append(tag)
                    self.sft_trg.append(soft_labels_trg)

                nbr += 1

        msg = "We expect to find {} samples to move to L', " \
              "but we found {}.".format(len(ids_to_move), nbr)
        assert nbr == len(ids_to_move), msg

    def change_u_to_lp(self, ids):
        """
        Change within this dataset the samples identified by the provided ids
        into L`. These samples are supposed to be in U.
        :param ids: list of ids of the concerned samples.
        :return:
        """
        msg = "`ids_to_move` is empty."
        assert ids, msg

        nbr = 0
        for i, sample in enumerate(self.samples):
            idx = sample[self.position['id']]
            if idx in ids:
                tag = self.samples[i][self.position['tag']]
                msg = "We expect the sample with the id {} to be unlabeled " \
                      "(tag = 1), but found tag = {}.".format(idx, tag)
                assert tag == constants.U, msg
                self.samples[i][self.position['tag']] = constants.PL  # move it.
                nbr += 1

        msg = "We expect to find {} samples to move to L', " \
              "but we found {}.".format(len(ids), nbr)
        assert nbr == len(ids), msg

    def set_up_new_seeds(self):
        """
        Set up new seed for each sample.
        :return:
        """
        self.seeds = self.get_new_seeds()

    def get_new_seeds(self):
        """
        Generate a seed per sample.
        :return: a list of random int.
        """
        return np.random.randint(0, 100000, len(self)).tolist()

    def get_original_input_img(self, i):
        """
        Returns the original input image read from disc.
        :param i: index of the sample.
        :return:
        """
        if self.dataset_name == 'mnist':  # grey image
            return Image.open(
                self.samples[i][self.position['img']], "r").convert("L")
        else:  # images with color
            return Image.open(
                self.samples[i][self.position['img']], "r").convert("RGB")

    def get_soft_trg_input(self, i):
        """
        Returns the soft target of the sample `i` read from disc.
        :param i: index of the sample.
        :return: torch.Tensor, the soft target of the sample `i`.
        """
        if self.soft_labels_trg[i] is None:
            return None
        with open(self.soft_labels_trg[i], "rb") as fpkl:
            return pkl.load(fpkl)

    def get_id_input(self, i):
        """
        Return the id of the sample `i`.
        :param i: int, index of the sample.
        :return: str, the id of the input sample.
        """
        return self.samples[i][self.position["id"]]

    def get_tag_input(self, i):
        """
        Return the tag of the sample `i`.
        :param i: int, index of the sample.
        :return: int, the tag of the input sample: {0, 1, 2}.
        """
        tag = int(self.samples[i][self.position["tag"]])
        msg = "tag {} is not the list of supported tags {}.".format(
            tag, self.possible_tags)

        assert tag in self.possible_tags, msg
        return tag

    def get_path_input_img(self, i):
        """
        Return the path of the input image.
        :param i: int, index of the sample.
        :return: str, path of the input image.
        """
        return self.samples[i][self.position["img"]]

    def get_original_input_mask(self, i):
        """
        Returns the original input mask read from disc.
        If the dataset does not have masks, we return None.
        :param i: index of the sample.
        :return: PIL.Image.Image mask.
        """
        id_s = self.samples[i][self.position['id']]
        tag_s = self.samples[i][self.position['tag']]

        # RULES: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #
        #
        # 1. ONLY THE TAG SAMPLE DETERMINES WHAT MASK SHOULD WE USE (
        # TRUE OR PSEUDO-MASK).
        #
        #
        # 2. THE TRUE MASK IS RETURNED ONLY IF THE TAG IS CONSTANTS.L OR
        # CONSTANTS.U. IN THE LATER CASE, THE MASK MUST NOT BE USED FOR
        # LEARNING. IT IT PROVIDED SO THE CODE DOES NOT BREAK.
        #
        #
        # 3. THE PSEUDO-MASK IS USED ONLY WHEN THE TAG IS CONSTANTS.PL. IN
        # SUCH CASE, THE `previous_pairs` IS USED TO GET THE PATH OF THE
        # PSEUDO-MASK WHERE THE SAMPLE'S ID MUST BE A KEY OF
        # `previous_pairs`. `self.check_ps_msk_path` MUST BE TRUE TO
        # EXPLICITLY ALLOW USING PSEUDO-MASKS.
        #
        #
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if tag_s in [constants.L, constants.U]:
            path = self.samples[i][self.position['mask']]
        elif tag_s == constants.PL:
            # sanity check.
            cnd_lp = id_s in self.previous_pairs.keys()
            msg = "'id':{} could not be found in " \
                  "`previous_pairs.keys()`.".format(id_s)
            assert cnd_lp, msg

            msg = "'self.check_ps_msk_path' is False while we found " \
                  "a sample with flag {}. If you want to use pseudo-masks, " \
                  "you need to explicitly set `self.check_ps_msk_path` " \
                  "to True. Set `self.fd_p_msks`."
            assert self.check_ps_msk_path, msg

            path = join(
                self.fd_p_msks, "{}-{}.bmp".format(
                    id_s, self.previous_pairs[id_s])
            )
            msg = "'path'={} does not exist.".format(path)
            assert os.path.exists(path), msg
        else:
            raise ValueError("unknown sample tag {}.".format(tag_s))

        # use the true mask to see if the perf. improves with the true labels.
        # path = self.samples[i][self.position['mask']]

        # None is replaced with ''.
        # https://docs.python.org/3/library/csv.html#csv.writer
        if (path is None) or (path == ''):
            return None

        mask = Image.open(path, "r").convert("L")

        # Caltech-UCSD-Birds-200-2011: a pixel belongs to the mask if its
        # value > 255/2. (an image is annotated
        # by many workers. If more than half of the workers agreed on the pixel
        # to be a bird, we consider that pixel as a bird.

        # Oxford-flowers-102: a pixel belongs to the mask if its value > 0.
        # The mask has only {0, 255} as values. The new binary mask will
        # contain only {0, 1} values where 0 is the background and 1 is the
        # foreground..
        mask_np = np.array(mask)
        if self.dataset_name == constants.GLAS:
            mask_np = (mask_np != 0).astype(np.uint8)
        elif self.dataset_name == constants.CUB:
            mask_np = (mask_np > (255 / 2.)).astype(np.uint8)
        elif self.dataset_name == constants.OXF:
            mask_np = (mask_np != 0).astype(np.uint8)
        elif self.dataset_name == constants.CAM16:
            mask_np = (mask_np != 0).astype(np.uint8)
        else:
            raise ValueError("Dataset name {} unsupported. The dataset "
                             "may not have the masks. If you think we "
                             "are wrong, come here and fix this. Exiting "
                             ".... [NOT OK]".format(self.dataset_name))

        mask = Image.fromarray(mask_np * 255, mode="L")

        return mask

    def get_original_input_label_int(self, i):
        """
        Returns the input image-level label as int.
        :param i: index of the sample.
        :return: int, the class of the sample as int.
        """
        label_str = self.samples[i][self.position['label']]
        return self.name_classes[label_str]

    def get_original_input_label_str(self, i):
        """
        Returns the input image-level label (the name of the class [str]).
        :param i: index of the sample.
        :return: int, the class of the sample as str.
        """
        return self.samples[i][self.position['label']]

    def load_sample_i(self, i, force_enhance_color=None):
        """
        Read from disc sample number i.
        :param i: index of the sample to load.
        :param force_enhance_color: bool, or None. if bool, we override the
        value of self.enhance_color. if None, nothing is done.
        :return: id, path_img, path_mask, label, tag
        """
        img = self.get_original_input_img(i)
        mask = self.get_original_input_mask(i)
        label = self.get_original_input_label_int(i)
        tag = self.get_tag_input(i)
        id_ = self.get_id_input(i)

        # convert mask into binary values. now it has values in {0, 255}.
        # the mask needs to have unique values {0, 1} so it wont be altered
        # when doing resizing. this has to be done in
        # self.get_original_input_mask. but, it is too late now.
        if mask is not None:
            mask_np = np.array(mask)
            mask_np = (mask_np != 0).astype(np.uint8)
            mask = Image.fromarray(mask_np, mode="L")

        self.original_images_size[i] = img.size

        enhance_color = self.enhance_color
        if isinstance(force_enhance_color, bool):
            enhance_color = enhance_color and force_enhance_color

        if enhance_color:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.enhance_color_fact)

        if self.resize:
            img = img.resize(self.resize)
            if self.resize_mask:
                msg = "mask is None and you asked for resizing the mask."
                assert mask is not None, msg
                msg = "mask's size {} mismatches the images' size {}." \
                      " expected to be the same.".format(
                       mask.size, self.original_images_size[i])
                assert mask.size == self.original_images_size[i], msg

                mask = mask.resize(self.resize)
        elif self.resize_h_to:
            w, h = img.size
            w_new, h_new = self.compute_proportional_size(w, h)
            img = img.resize((w_new, h_new))
            if self.resize_mask:
                msg = "mask is None and you asked for resizing the mask."
                assert mask is not None, msg
                msg = "mask's size {} mismatches the images' size {}." \
                      " expected to be the same.".format(
                       mask.size, self.original_images_size[i])
                assert mask.size == self.original_images_size[i], msg
                mask = mask.resize((w_new, h_new))

        return id_, img, mask, label, tag

    def compute_proportional_size(self, w, h):
        """
        Compute the proportional size when rescaling the height into
        'self.resize_h_to'.

        :param w: int. the current width.
        :param h: int. the current height.
        :return: w`, h` where h` == self.resize_h_to, and w` is the new width
        that preserves the proportional scaling from h to h`.
        """
        msg = "'self.resize_h_to' is none. it should not be the case."
        assert self.resize_h_to is not None, msg

        r = self.resize_h_to / float(h)
        w_new = int(r * w)

        return w_new, self.resize_h_to

    @staticmethod
    def compute_proportional_size_stat(w, h, resize_h_to):
        """
        Compute the proportional size when rescaling the height into
        'resize_h_to'.
        same as self.compute_proportional_size.

        :param w: int. the current width.
        :param h: int. the current height.
        :param resize_h_to: int. the desired height.
        :return: w`, h` where h` == resize_h_to, and w` is the new width
        that preserves the proportional scaling from h to h`.
        """


        r = resize_h_to / float(h)
        w_new = int(r * w)

        return w_new, resize_h_to

    def tmp_update_soft_lab_with_id(self, soft_labels_trg, id_):
        """
        Update the soft label target associated with the id.
        :param soft_labels_trg: torch vector of the new temporal probability.
        :param id_: float, the id to which make the update.
        :return:
        """
        msg = "'soft_labels_trg' must be an instance of torch.tensor. Found " \
              "{}".format(type(soft_labels_trg))
        assert isinstance(soft_labels_trg, torch.Tensor), msg

        idx = None
        for i in range(len(self)):
            if id_ == self.get_id_input(i):
                idx = i
                break
        if idx is None:
            raise ValueError("Something is wrong. We have scanned the entire"
                             "dataset, but we didnt find a sample with the "
                             "specified id {}".format(id_))

        msg = "You asked to update the soft target label but it seems that " \
              "this dataset does not expect any soft target since it is None."
        assert self.soft_labels_trg[idx] is not None, msg

        with open(self.soft_labels_trg[idx], "wb") as fpkl:
            pkl.dump(soft_labels_trg, fpkl, protocol=pkl.HIGHEST_PROTOCOL)

    def get_index_of_id(self, id_):
        """
        Return the index of the id.
        :param id_: str, id of the sample.
        """
        idx = None
        # TODO: in a normal setup, this can be done way better by avoiding
        #  the loop by storing once the ids in self.__init__(). then,
        #  use builtin function list.index().
        # for i in range(len(self)):
        #     if id_ == self.get_id_input(i):
        #         idx = i
        #         break
        try:
            idx = self.list_ids.index(id_)
        except ValueError as error:
            print(error)
            raise ValueError("Something is wrong. We have scanned the entire"
                             "dataset, but we didn't find a sample with the "
                             "specified id {}".format(id_))

        return idx

    def preload_images(self):
        """
        Preload images/masks[optional]/labels.
        :return:
        """
        print("Preloading the images of `{}` dataset. "
              "This may take some time ... [OK]".format(self.dataset_name))
        tx = dt.datetime.now()

        for i in tqdm.tqdm(range(self.n), ncols=80, total=self.n):
            id_, img, mask, label, tag = self.load_sample_i(i)

            self.images.append(img)
            self.masks.append(mask)
            self.labels.append(label)
            self.ids.append(id_)
            self.tags.append(tag)

        self.preloaded = True
        print("{} has successfully loaded the images with {} samples in"
              " {} .... [OK]".format(
               self.dataset_name, self.n, dt.datetime.now() - tx))

    @staticmethod
    def get_upscaled_dims(w, h, up_scale_small_dim_to):
        """
        Compute the upscaled dimensions using the size `up_scale_small_dim_to`.

        :param w:
        :param h:
        :param up_scale_small_dim_to:
        :return: w, h: the width and the height upscale (with preservation of
         the ratio).
        """
        if up_scale_small_dim_to is None:
            return w, h

        s = up_scale_small_dim_to
        if h < s:
            if h < w:  # find the maximum ratio to scale.
                r = (s / h)
            else:
                r = (s / w)
        elif w < s:  # find the maximum ratio to scale.
            if w < h:
                r = (s / w)
            else:
                r = (s / h)
        else:
            r = 1  # no upscaling since both dims are higher or equal to
            # the min (s).
        h_, w_ = int(h * r), int(w * r)

        return w_, h_

    def set_ready_eval_fn(self):
        """
        Prepare the data for evaluation [Called ONLY ONCE].

        This function is useful when this class is instantiated over an
        evaluation set with no randomness, such as the valid set or the test
        set.

        The idea is to prepare the data by performing all the necessary steps
        until we arrive to the final form of the input of the model.

        This will avoid doing all the steps every time self.__getitem__() is
        called.

        :return:
        """
        assert self.set_for_eval, "Something wrong. You didn't ask to set " \
                                  "the data ready for evaluation, " \
                                  "but here we are .... [NOT OK]"
        assert self.images, "self.images is not ready yet. " \
                            "Re-check .... [NOT OK]"
        assert self.masks, "self.masks is not ready yet. " \
                           "Re-check ... [NOT OK]"
        assert self.labels, "self.labels is not ready yet. " \
                            "Re-check ... [NOT OK]"
        assert self.ids, "self.ids is not ready yet. " \
                         "Re-check ... [NOT OK]"
        assert self.tags, "self.tags is not ready yet. " \
                          "Re-check ... [NOT OK]"

        print("Setting `{}` this dataset for evaluation. "
              "This may take some time ... [OK]".format(self.dataset_name))
        tx = dt.datetime.now()

        # Turn off momentarily self.set_for_eval.
        self.set_for_eval = False

        for i in tqdm.tqdm(range(len(self.images)), ncols=80, total=self.n):
            id_, img, mask, label, tag = self.__getitem__(i)
            self.inputs_ready.append(img)
            self.masks_ready.append(mask)
            self.labels_ready.append(label)
            self.tags_ready.append(tag)
            self.ids_ready.append(id_)

        # Turn self.set_for_eval back on.
        self.set_for_eval = True
        # Now that we preloaded everything, we need to remove self.images,
        # self.masks, to preserve space!!!
        # We keep self.labels. We need it!!! and it does not take much space!
        del self.images
        del self.masks
        del self.labels

        print("`{}` dataset has been set ready for evaluation with"
              " `{}` samples ready to go in {} .... [OK]".format(
                self.dataset_name, self.n, dt.datetime.now() - tx))

    @staticmethod
    def get_padding(s, c):
        """
        Find out how much padding in both sides (left/right) or (top/bottom)
        is required.
        :param s: height or width of the image.
        :param c: constant such as after padding we will have: s % c = 0.
        :return: pad1, pad2. Padding in both sides.
        """
        assert isinstance(s, int) and isinstance(c, int), "s, and c must " \
                                                          "be integers ...." \
                                                          " [NOT OK]"

        if s % c == 0:
            return 0, 0
        leftover = c - s % c
        if leftover % 2 == 0:
            return int(leftover / 2), int(leftover / 2)
        else:
            return int(leftover / 2), leftover - int(leftover / 2)

    def __getitem__(self, index, crop_cord=None):
        """
        Return one sample and its label and extra information that we need
        later.

        :param index: int, the index of the sample within the whole dataset.
        :param crop_cord: tuple (ii, j, h, w), the cropping coordinates that
        are forced instead of the ones used by the class.
        :return:
            id_, img, mask, label, tag, soft_labels_trg. where:
            id_: float, the id of the sample.
            img: torch.Tensor of sixe (1, C, H, W), the input image.
            mask: PIL.Image.Image, the mask of the regions of interest, or None.
            label: int, the class label of the sample.
            tag: int, the sample tag.
            soft_labels_trg: torch.Tensor vector. The soft label target of
            the sample.
            out_crop_cord: tuple. if there has been a cropping, return (i, j,
            h, w), the cropping coordinates. otherwise, returns None.
        """
        # Force seeding: a workaround to deal with reproducibility when suing
        # different number of workers if want to
        # preserve the reproducibility. Each sample has its won seed.
        reproducibility.reset_seed(self.seeds[index])

        # todo: URGENT: improve the dataset (better cropping with more data
        #  augmentation.) [one day in the far? future.]

        if self.set_for_eval:
            error_msg = "Something wrong. You didn't ask to set the data " \
                        "ready for evaluation, but here we are " \
                        ".... [NOT OK]"
            cond = self.inputs_ready != []
            cond = cond and (self.labels_ready != [])
            cond = cond and (self.masks_ready != [])
            cond = cond and (self.tags_ready != [])
            cond = cond and (self.ids_ready != [])
            assert cond, error_msg

            img = self.inputs_ready[index]
            mask = self.masks_ready[index]
            label = self.labels_ready[index]
            tag = self.tags_ready[index]
            id_ = self.ids_ready[index]

            return id_, img, mask, label, tag

        if self.do_not_save_samples:
            id_, img, mask, label, tag = self.load_sample_i(index)
        else:
            assert self.preloaded, "Sorry, you need to preload the data " \
                                   "first .... [NOT OK]"
            id_ = self.ids[index]
            img = self.images[index]
            mask = self.masks[index]
            label = self.labels[index]
            tag = self.tags[index]

        # we do the following transformations in this order:
        # 1. upscale min_dim
        # 2. padding
        # 3. [crop]
        # 4. ratio_scale.

        # Upscale on the fly. Sorry, this may add an extra time, but, we do
        # not want to save in memory upscaled images!!!! it takes a lot of
        # space, especially for large datasets. So, compromise? upscale only
        # when necessary.
        # check if we need to upscale the image.
        # Useful for Caltech-UCSD-Birds-200-2011 for instance.
        if self.up_scale_small_dim_to is not None:
            w, h = img.size
            w_up, h_up = self.get_upscaled_dims(
                w, h, self.up_scale_small_dim_to)
            # make a resized copy.
            img = img.resize((w_up, h_up), resample=self.scale_algo)
            # resize the mask as well. needed for the training only.
            # for the evaluation, we keep the original mask.
            # this operation will not alter the unique values in the binary
            # mask {0, 1}.
            if (mask is not None) and (not self.for_eval_flag):
                mask = mask.resize((w_up, h_up), resample=self.scale_algo)

        # Upscale the image: only for Caltech-UCSD-Birds-200-2011 and similar
        # datasets.

        # Padding.
        if self.padding_size:
            w, h = img.size
            ph, pw = self.padding_size
            padding = (int(pw * w), int(ph * h))
            img = TF.pad(img, padding=padding,
                         padding_mode=self.padding_mode)
            # just for training.
            if (mask is not None) and (not self.for_eval_flag):
                mask = TF.pad(mask, padding=padding,
                              padding_mode=self.padding_mode)

        # crop a patch (training only). Do not crop for evaluation.
        out_crop_cord = None
        if self.randomCropper or (crop_cord is not None):
            if crop_cord is None:
                msg = "Something's wrong. This is expected to be False." \
                      "We do not crop for evaluation."
                assert not self.for_eval_flag, msg
            if self.randomCropper:
                img, (i, j, h, w) = self.randomCropper(img)
                out_crop_cord = (i, j, h, w)
                # print("INDEX {} i {} j {} h {} w {} seed {}".format(
                #     index, i, j, h, w,self.seeds[index]))
            elif crop_cord is not None:
                (i, j, h, w) = crop_cord
                img = TF.crop(img, i, j, h, w)
                out_crop_cord = crop_cord
            else:
                raise ValueError("Something is wrong. Leaving...")
            # print("Dadaloader Index {} i  {}  j {} seed {}".format(
            # index, i, j, self.seeds[index]))
            # crop the mask just for tracking. Not used for actual training.
            if mask is not None:
                mask = TF.crop(mask, i, j, h, w)

            if self.ratio_scale_patch < 1.:
                img = img.resize((int(img.size[0] * self.ratio_scale_patch),
                                  int(img.size[1] * self.ratio_scale_patch))
                                 )

                if mask is not None:  # resize the mask as well. (h, w) and
                    # not (w, h).
                    uniqvals_before = np.unique(np.asanyarray(mask))
                    mask = TF.resize(
                        mask, size=(int(mask.size[1] * self.ratio_scale_patch),
                                    int(mask.size[0] * self.ratio_scale_patch)),
                        interpolation=self.scale_algo)
                    uniqvals_after = np.unique(np.asanyarray(mask))
                    # basically, the unique values should not altered. but,
                    # just in case.
                    msg = "interpolating the mask has altered the unique " \
                          "values. this was unexpected. fix it." \
                          "before {}, after {}.".format(
                            uniqvals_before, uniqvals_after)

                    # assert all(uniqvals_after == uniqvals_before), msg

        # rescale the image with the same ration that we use to rescale the
        # cropped patches.
        if self.for_eval_flag and (self.ratio_scale_patch < 1.):
            img = img.resize((int(img.size[0] * self.ratio_scale_patch),
                              int(img.size[1] * self.ratio_scale_patch)))

        # just for training: do not transform the mask (since it is not used).
        if self.transform_img:
            img = self.transform_img(img)

        # just for training: do not transform the mask (since it is not used).
        if self.transform_tensor:
            img = self.transform_tensor(img)

        # resize tensor into 32x32 for mnist dataset
        if self.dataset_name == constants.MNIST:
            tmp = torch.zeros((32, 32), dtype=img.dtype)
            tmp[2:30, 2:30] = img
            img = tmp
            img = img.view(1, 32, 32)

        # Prepare the mask to be used on GPU to compute Dice index.
        if mask is not None:
            mask = np.array(mask, dtype=np.float32)  # full of 0
            # and 1. mak the mask with shape (h, w, 1), then to_tensor will
            # convert it into (1, h, w).
            mask = self.to_tensor(np.expand_dims(mask, axis=-1))

            # msg = "height/width mismatch. img {}. mask {}. id {}".format(
            #     img.shape, mask.shape, id_
            # )
            # assert img.shape[1:] == mask.shape[1:], msg
        # ======================================================================
        #              IF MASK IS NONE: CASE OF CAMELYON16, WE RETURN
        #       AN EMPTY MASK FULL OF ZERO. THIS HELPS IN NEXT OPERATIONS.
        # shape: (1, height_img, width_imag).
        # dtype: torch.float.
        # ======================================================================
        else:
            msg = "'None' masks are supported and expected only for CAM16" \
                  "dataset. Current dataset: {}. (ERROR)".format(
                self.dataset_name)
            assert self.dataset_name == constants.CAM16, msg

            mask = torch.zeros((1, img.shape[1], img.shape[2]),
                               dtype=torch.float, requires_grad=False)

        return id_, img, mask, label, tag, out_crop_cord

    def turnback_mask_tensor_into_original_size(self, pred_masks, oh, ow):
        """
        Resize the mask into its original size the right way.
        This is helpful during evaluation (validation/test and not training).
        an error will be raise if the dataset is not flagged for validation.

        Note:
        In this dataset, the produced image has gone through the following
        ordered transformations. it is these transformations that had led to
        the size of the mask (pred_mask). this assumes that the predicted
        mask has the same size as the input image of the network (i.e.,
        the image produced by this dataset).
        1. resize, or resize_h_to. [not supported for now.]
        2. up_scale_small_dim_to.
        3. padding_size.
        4. ratio_scale_patch.

        to get the correct size of the mask, we need to reverse these
        operations from the last one to the first one.

        :param pred_masks: pytorch tensor of shape (b, p, h`, w`). predicted
        tensor.
        :param oh: int. original height os the mask.
        :param ow: oroginal width of the mask.

        :return: resized mask with shape (b, p, oh, ow).
        """
        msg = "this function is made only for datasets that has been set for " \
              "vlaidation 'for_eval_flag' is true. found " \
              "'for_eval_flag'={}. unsupported.".format(self.for_eval_flag)
        assert self.for_eval_flag, msg
        msg = "'resize' must be None. found {}. unsupported".format(self.resize)
        assert self.resize is None, msg
        msg = "'resize_h_to' must be None. found {}. unsupported".format(
            self.resize_h_to)
        assert self.resize_h_to is None, msg

        msg = "'pred_masks,ndim' must be 4. found {}.".format(pred_masks.ndim)
        assert pred_masks.ndim == 4, msg

        # 4. reverse ratio scale.
        if self.ratio_scale_patch < 1.:
            _, _, h, w = pred_masks.shape
            hr = int(h / self.ratio_scale_patch)
            wr = int(w / self.ratio_scale_patch)
            pred_masks = F.interpolate(pred_masks,
                                       size=(hr, wr),
                                       mode='bilinear',
                                       align_corners=False
                                       )

        # 3. reverse padding.
        if self.padding_size:
            _, _, h, w = pred_masks.shape
            ph, pw = self.padding_size
            pdh = - int(ph * h)  # negative padding (top=bottom)
            pdw = - int(ph * w)  # negative padding. (left=right)
            pred_masks = F.pad(input=pred_masks, pad=(pdw, pdw, pdh, pdh))

        # 2. reverse up_scale_small_dim_to
        if self.up_scale_small_dim_to is not None:
            pred_masks = F.interpolate(pred_masks,
                                       size=(oh, ow),
                                       mode='bilinear',
                                       align_corners=False
                                       )

        # by now, the size of the predicted mask is the same as the original
        # one. but, few pixels may have been added or missed due to rounding.
        pred_masks = F.interpolate(pred_masks,
                                   size=(oh, ow),
                                   mode='bilinear',
                                   align_corners=False
                                   )

        return pred_masks

    def __len__(self):
        return len(self.samples)
