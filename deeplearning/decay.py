import numpy as np
import matplotlib.pyplot as plt


__all__ = ['ConstantWeight', 'LinearAnnealedWeight', 'ExponentialDecayWeight']


class Decay(object):
    """
    Parent class for decay.
    """
    def __init__(self, init_val, end_val, max_epochs, sigma):
        """
        Init function.
        :param init_val: float. initial value of the weight.
        :param end_val: float. final value (or minimum value allowed).
        :param max_epochs: int. maximum number of epochs.
        :param sigma: float. scaling factor to change the rate of the curve.
        the higher the value, the slower the slope.
        """
        pass

    def __call__(self):
        """
        Update the weight according to the predefine schedule.
        """
        raise NotImplementedError

    def get_current_weight(self):
        """
        Return the current value of the weight.
        """
        raise NotImplementedError
    def __repr__(self):
        """
        rep of the class.
        """
        raise NotImplementedError


class ConstantWeight(Decay):
    """
    A callback to adjust the weight.
    Schedule: keep the weight fixed to some specific value.
    """
    def __init__(self, init_val, end_val, max_epochs, sigma):
        """
        Init. function.
        :param init_val: float. initial value of the weight.
        :param end_val: float. final value (or minimum value allowed).
        :param max_epochs: int. maximum number of epochs.
        :param sigma: float. scaling factor to change the rate of the curve.
        the higher the value, the slower the slope.
        """
        super(ConstantWeight, self).__init__(
            init_val, end_val, max_epochs, sigma)
        self._init_val = init_val

    def __call__(self):
        """
        Update the weight according the annealing schedule.
        """
        return self.get_current_weight()

    def get_current_weight(self):
        """
        Calculate the current weight according to the annealing
        schedule.
        """
        return self._init_val

    def __repr__(self):
        return "{}(init_val={})".format(
            self.__class__.__name__, self._init_val
        )



class LinearAnnealedWeight(Decay):
    """
    A callback to adjust a weight linearly.

    Linearly anneal a weight from init_value to end_value.
    w(t) := w(t-1) - rate.
    where:
    rate := (init_value - end_value) / max_epochs.

    the scale is computed based on the maximum epochs.

    ref:
    S. Belharbi, R. Hérault, C. Chatelain and S. Adam,
    “Deep multi-task learning with evolving weights”, in European Symposium
    on Artificial Neural Networks (ESANN), 2016.
    """
    def __init__(self, init_val, end_val, max_epochs, sigma):
        """
        Init. function.
        :param init_val: float. initial value of the weight.
        :param end_val: float. final value (or minimum value allowed).
        :param max_epochs: int. maximum number of epochs.
        :param sigma: float. scaling factor to change the rate of the curve.
        the higher the value, the slower the slope.
        """
        super(LinearAnnealedWeight, self).__init__(
            init_val, end_val, max_epochs, sigma)

        self._count = 0.
        self._anneal_start = init_val
        self._anneal_end = end_val
        msg = "'init_val' must be >= 'end_val'"
        assert init_val >= end_val, msg
        self._max_epochs = max_epochs
        self.anneal_rate = (init_val - end_val) / float(max_epochs)

        self.weight = init_val  # holds the current value.
        self._count = 0

    def __call__(self):
        """
        Updates the weight according to the annealing schedule.

        return: float. the new updated value.
        """
        if self._count == 0:
            self._count += 1
            return self.weight  # return the initial value.
        else:
            return self.get_current_weight()

    def get_current_weight(self):
        """
        Calculate the current weight according to the annealing
        schedule.
        """
        self.weight = self.weight - self.anneal_rate
        return max(self._anneal_end, self.weight)

    def __repr__(self):
        return "{}(init_val={}, end_val={}, max_epochs={})".format(
            self.__class__.__name__, self._anneal_start, self._anneal_end,
            self._max_epochs
        )


class ExponentialDecayWeight(Decay):
    """
    This anneals the weight exponentially with respect to the current epoch.

    w(t) = exp(-t/sigma).

    where `t` is the current epoch number, and `sigma`, is a constant that
    affects the slope of the function.

    ref:
    S. Belharbi, R. Hérault, C. Chatelain and S. Adam,
    “Deep multi-task learning with evolving weights”, in European Symposium
    on Artificial Neural Networks (ESANN), 2016.
    """
    def __init__(self, init_val, end_val, max_epochs, sigma):
        """
        Init. function.
        :param end_val: float. minimal value allowed.
        :param sigma: float. scaling factor to change the rate of the curve.
        the higher the value, the slower the slope.
        """
        super(ExponentialDecayWeight, self).__init__(
            init_val, end_val, max_epochs, sigma)

        self._count = 0
        assert sigma != 0, "'sigma'=0. must be different than zero."

        self._sigma = float(sigma)
        self._end_val = end_val

        self.weight = self.get_current_weight()

    def __call__(self):
        """Update the learning rate according to the exponential decay
        schedule.

        """
        if self._count == 0:
            self._count += 1
            return self.weight  # return the initial value.
        else:
            return self.get_current_weight()

    def get_current_weight(self):
        """
        Calculate the current weight according to the annealing
        schedule.
        """
        self.weight = np.exp(- self._count / float(self._sigma))
        self._count += 1

        return max(self.weight, self._end_val)

    def __repr__(self):
        return "{}(end_val={}, sigma={})".format(
            self.__class__.__name__, self._end_val, self._sigma
        )

if __name__ == "__main__":
    init_val, end_val, max_epochs, sigma = 1., 0.0001, 1000, 150.

    instances = [ConstantWeight(init_val, end_val, max_epochs, sigma),
                 LinearAnnealedWeight(init_val, end_val, max_epochs, sigma),
                 ExponentialDecayWeight(init_val, end_val, max_epochs, sigma)
                 ]
    fig = plt.figure()
    for inst in instances:
        plt.plot(
            [inst() for _ in range(max_epochs)],
            label=str(inst)
        )
    plt.legend(loc="lower right")
    fig.savefig('test-decay.png')
    plt.close('all')

