from collections import Sequence
import warnings
import copy

from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler


from deeplearning import models_cl, models_seg, criteria
from deeplearning import lr_scheduler as my_lr_scheduler
from deeplearning import sampling

from tools import Dict2Obj, count_nb_params


import constants


def instantiate_sampler(args, device):
    """
    Instantiate the sampler.

    :param args: Dict2Obj. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: sampler: instance of deeplearning.sampling.Sampler.
    """
    return sampling.Sampler(al_type=args.al_type,
                            p_samples=args.p_samples,
                            p_init_samples=args.p_init_samples,
                            device=device,
                            task=args.task
                            )


def instantiate_loss(args):
    """
    Instantiate the train loss.
    Can be used as well for evaluation.

    :param args: Dict2Obj. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: train_loss: instance of deeplearning.criteria.py losses.
    """
    loss_str = args.loss
    msg = "Unknown loss {}. It must be in [{}]" \
          " ... [NOT OK]".format(loss_str, '--'.join(constants.losses))

    assert loss_str in constants.losses, msg

    if args.task == constants.CL:
        if args.al_type == constants.AL_LP:
            assert args.loss == constants.KL, "something is wrong"

        if loss_str == constants.CE:
            return criteria.__dict__[args.loss]()
        elif loss_str == constants.KL:
            return criteria.__dict__[args.loss]()

    elif args.task == constants.SEG:
        return criteria.__dict__[args.loss](
            segloss_l=args.segloss_l,
            segloss_pl=args.segloss_pl,
            smooth=args.seg_smooth,
            elbon=args.seg_elbon,
            init_t=args.seg_init_t,
            max_t=args.seg_max_t,
            mulcoef=args.seg_mulcoef,
            subtask=args.subtask,
            scale_cl=args.scale_cl,
            scale_seg=args.scale_seg,
            scale_seg_u=args.scale_seg_u,
            scale_seg_u_end=args.scale_seg_u_end,
            scale_seg_u_sigma=args.scale_seg_u_sigma,
            scale_seg_u_sch=args.scale_seg_u_sch,
            max_epochs=args.max_epochs,
            freeze_classifier=args.freeze_classifier,
            weight_pl=1.
        )
    else:
        raise ValueError('Unknown task {}'.format(args.task))

    raise ValueError("Something's wrong. We do not even know how you "
                     "reached this line. Loss_str: {}"
                     " .... [NOT OK]. EXITING".format(loss_str))


def instantiate_models(args, verbose=True):
    """Instantiate the necessary models.
    Input:
        args: Dict2Obj. Contains the configuration of the exp that has been read
        from the yaml file.
    Output:
        instance of a model
    """
    p = Dict2Obj(args.model)
    if args.task == constants.CL:
        if p.name_model == constants.LENET5:
            model = models_cl.__dict__[p.name_model](
                num_classes=args.num_classes)
        elif p.name_model == constants.SOTASSL:
            model = models_cl.__dict__[p.name_model](
                num_classes=args.num_classes, dropoutnetssl=p.dropoutnetssl,
                modalities=p.modalities, kmax=p.kmax,
                kmin=p.kmin, alpha=p.alpha, dropout=p.dropout)
        else:
            raise ValueError("Unsupported model name: {}.".format(p.name_model))
    elif args.task == constants.SEG:
        if p.name_model == 'hybrid_model':
            model = models_seg.__dict__[p.name_model](
                num_classes=args.num_classes, num_masks=args.num_masks,
                backbone=p.backbone, pretrained=p.pre_trained,
                modalities=p.modalities, kmax=p.kmax,
                kmin=p.kmin, alpha=p.alpha, dropout=p.dropout,
                backbone_dropout=p.backbone_dropout,
                freeze_classifier=args.freeze_classifier,
                base_width=p.base_width, leak=p.leak
            )
        else:
            raise ValueError("Unknown model name for SEG task: {}".format(
                p.name_model))
    else:
        raise ValueError("Unknown task {}.".format(args.task))

    if verbose:
        print("`{}` was successfully instantiated. "
              "Nbr.params: {} .... [OK]".format(model, count_nb_params(model)))
    return model


def standardize_otpmizers_params(optm_dict):
    """
    Standardize the keys of a dict for the optimizer.
    all the keys starts with 'optn[?]__key' where we keep only the key and
    delete the initial.
    the dict should not have a key that has a dict as value. we do not deal
    with this case. an error will be raise.

    :param optm_dict: dict with specific keys.
    :return: a copy of optm_dict with standardized keys.
    """
    msg = "'optm_dict' must be of type dict. found {}.".format(type(optm_dict))
    assert isinstance(optm_dict, dict), msg
    new_optm_dict = copy.deepcopy(optm_dict)
    loldkeys = list(new_optm_dict.keys())

    for k in loldkeys:
        if k.startswith('optn'):
            msg = "'{}' is a dict. it must not be the case." \
                  "otherwise, we have to do a recursive thing....".format(k)
            assert not isinstance(new_optm_dict[k], dict), msg

            new_k = k.split('__')[1]
            new_optm_dict[new_k] = new_optm_dict.pop(k)

    return new_optm_dict


def instantiate_optimizer(args, model, verbose=True):
    """Instantiate an optimizer.
    Input:
        args: object. Contains the configuration of the exp that has been
        read from the yaml file.
        mode: a pytorch model with parameters.

    Output:
        optimizer: a pytorch optimizer.
        lrate_scheduler: a pytorch learning rate scheduler (or None).
    """
    params = copy.deepcopy(args.optimizer)
    params = standardize_otpmizers_params(params)
    params = Dict2Obj(params)

    if params.name_optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=params.lr,
                        momentum=params.momentum,
                        dampening=params.dampening,
                        weight_decay=params.weight_decay,
                        nesterov=params.nesterov)
    elif params.name_optimizer == "adam":
        optimizer = Adam(params=model.parameters(), lr=params.lr,
                         betas=(params.beta1, params.beta2),
                         eps=params.eps_adam,
                         weight_decay=params.weight_decay,
                         amsgrad=params.amsgrad)
    else:
        raise ValueError("Unsupported optimizer `{}` .... "
                         "[NOT OK]".format(args.optimizer["name"]))

    if verbose:
        print("Optimizer `{}` was successfully instantiated .... "
              "[OK]".format(
                [key + ":" + str(args.optimizer[key]) for
                    key in args.optimizer.keys()]))

    if params.lr_scheduler:
        if params.name_lr_scheduler == "step":
            lrate_scheduler = lr_scheduler.StepLR(
                optimizer, step_size=params.step_size,
                gamma=params.gamma,
                last_epoch=params.last_epoch)

        elif params.name_lr_scheduler == "cosine":
            lrate_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.t_max,
                eta_min=params.min_lr,
                last_epoch=params.last_epoch)

        elif params.name_lr_scheduler == "mystep":
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer, step_size=params.step_size,
                gamma=params.gamma,
                last_epoch=params.last_epoch,
                min_lr=params.min_lr)

        elif params.name_lr_scheduler == "mycosine":
            lrate_scheduler = my_lr_scheduler.MyCosineLR(
                optimizer, coef=params.coef,
                max_epochs=params.max_epochs,
                min_lr=params.min_lr,
                last_epoch=params.last_epoch)

        elif params.name_lr_scheduler == "multistep":
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=params.milestones,
                gamma=params.gamma,
                last_epoch=params.last_epoch)

        else:
            raise ValueError("Unsupported learning rate scheduler `{}` .... "
                             "[NOT OK]".format(
                                params.name_lr_scheduler))

        if verbose:
            print(
                "Learning scheduler `{}` was successfully "
                "instantiated "
                ".... [OK]".format(params.name_lr_scheduler))
    else:
        lrate_scheduler = None

    return optimizer, lrate_scheduler


def instantiate_updater_label_n_per_class(args):
    """
    Instantiate the class that update the number of samples to label at each
    epoch.

    :param args: Dict2Obj. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: instance: instance of a class inherited from
    deeplearning.criteria._UpdaterNBRUToLabel().
    """
    upperstr = args.updater_label_n_per_class
    msg = "Unknown name {}. It must be in [`UpdaterNBRUToLabelConstant`," \
          "`UpdaterNBRUToLabelConstant`]... [NOT OK]".format(upperstr)

    assert upperstr in ['UpdaterNBRUToLabelConstant',
                        'UpdaterNBRUToLabelIncremental'], msg
    if upperstr == "UpdaterNBRUToLabelConstant":
        return criteria.__dict__[args.updater_label_n_per_class](
            init_val=args.label_n_per_class, max_val=args.cap_label_n_per_class,
            epoch_start_label_unlabeled_samples=
            args.epoch_start_label_unlabeled_samples)
    elif upperstr == 'UpdaterNBRUToLabelIncremental':
        return criteria.__dict__[args.updater_label_n_per_class](
            init_val=args.label_n_per_class, max_val=args.cap_label_n_per_class,
            epoch_start_label_unlabeled_samples=
            args.epoch_start_label_unlabeled_samples,
            increment=args.increment_label_n_per_class)

    raise ValueError("Something's wrong. We do not even know how you "
                     "reached this line. Updater: {}"
                     " .... [NOT OK]. EXITING".format(upperstr))


def instantiate_temporal_updater(args):
    """
    Instantiate the class that does the temporal update.

    :param args: Dict2Obj. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: instance: instance of a class inherited from
    deeplearnig.criteria._TemporalUpdate().
    """
    name = args.temporal_updater
    support = ["TemporalUpdateMomentum", "TemporalUpdateAdam"]
    msg = "Unknown name. It must be in {}.".format(support)
    assert name in support, msg

    if name == 'TemporalUpdateMomentum':
        return criteria.__dict__[name](alpha=args.alpha_tmp)
    elif name == 'TemporalUpdateAdam':
        return criteria.__dict__[name]()


def instantiate_quantor(args):
    """
    Instantiate the class that select the best samples in U to move to L`.

    :param args: Dict2Obj. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: instance: instance of a class inherited from
    deeplearnig.criteria._Quantor().
    """
    name = args.temporal_updater
    support = ["EntropyQuantor", "ScoreQuantor"]
    msg = "Unknown name. It must be in {}.".format(support)
    assert name in support, msg

    if name == 'EntropyQuantor':
        return criteria.__dict__[name]()
    elif name == 'ScoreQuantor':
        return criteria.__dict__[name]()
