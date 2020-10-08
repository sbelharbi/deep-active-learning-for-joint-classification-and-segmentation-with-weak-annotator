import warnings
import math

import torch.optim.lr_scheduler as lr_scheduler


class MyStepLR(lr_scheduler.StepLR):
    """
    Override: https://pytorch.org/docs/1.0.0/_modules/torch/optim/lr_scheduler.html#StepLR
    Reason: we want to fix the learning rate to not get lower than some value:
    min_lr.

    Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        min_lr (float): The lowest allowed value for the learning rate.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1,
                 min_lr=1e-6):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(lr_scheduler.StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** (self.last_epoch // self.step_size),
                self.min_lr) for base_lr in self.base_lrs]


class MyCosineLR(lr_scheduler.StepLR):
    """
    Override: https://pytorch.org/docs/1.0.0/_modules/torch/optim/
    lr_scheduler.html#StepLR
    Reason: use a cosine evolution of lr.
    paper:
    `S. Qiao, W. Shen, Z. Zhang, B. Wang, and A. Yuille.  Deepco-training for
    semi-supervised image recognition. InECCV,2018`


    for the epoch T:
    lr = base_lr * coef × (1.0 + cos((T − 1) × π/max_epochs)).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        coef (float): float coefficient. e.g. 0.005
        max_epochs (int): maximum epochs.
        last_epoch (int): The index of last epoch. Default: -1.
        min_lr (float): The lowest allowed value for the learning rate.
    """

    def __init__(self, optimizer, coef, max_epochs, min_lr=1e-9,
                 last_epoch=-1):
        assert isinstance(coef, float), "'coef' must be a float. found {}" \
                                        "...[not ok]".format(type(coef))
        assert coef > 0., "'coef' must be > 0. found {} ....[NOT OK]".format(
            coef
        )
        assert max_epochs > 0, "'max_epochs' must be > 0. found {}" \
                               "...[NOT OK]".format(max_epochs)
        self.max_epochs = float(max_epochs)
        self.coef = coef
        self.min_lr = min_lr
        super(lr_scheduler.StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.coef * (
                    1. + math.cos((self.last_epoch - 1) * math.pi /
                                  self.max_epochs)),
                self.min_lr) for base_lr in self.base_lrs]


if __name__ == "__main__":
    from torch.optim import SGD
    import torch
    import matplotlib.pyplot as plt

    optimizer = SGD(torch.nn.Linear(10, 20).parameters(), lr=0.001)
    lr_sch = MyCosineLR(optimizer, coef=0.5, max_epochs=600, min_lr=1e-9)
    vals = []
    for i in range(1000):
        optimizer.step()
        vals.append(lr_sch.get_lr())
        lr_sch.step()
    plt.plot(vals)
    plt.savefig("lr_evolve_{}.png".format(lr_sch.__class__.__name__))
