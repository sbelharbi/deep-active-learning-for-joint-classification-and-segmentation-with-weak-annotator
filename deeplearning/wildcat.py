import sys


import torch.nn as nn

sys.path.append("..")

from deeplearning.decision_pooling import WildCatPoolDecision, ClassWisePooling


class WildCatClassifierHead(nn.Module):
    """
    A WILDCAT type classifier head.
    `WILDCAT: Weakly Supervised Learning of Deep ConvNets for
    Image Classification, Pointwise Localization and Segmentation`,
    Thibaut Durand, Taylor Mordan, Nicolas Thome, Matthieu Cord.
    """
    def __init__(self, inplans, modalities, num_classes, kmax=0.5, kmin=None,
                 alpha=0.6, dropout=0.0):
        super(WildCatClassifierHead, self).__init__()
        self.name = "wildcat"

        self.num_classes = num_classes

        self.to_modalities = nn.Conv2d(
            inplans, num_classes * modalities, kernel_size=1, bias=True)
        self.to_maps = ClassWisePooling(num_classes, modalities)
        self.wildcat = WildCatPoolDecision(
            kmax=kmax, kmin=kmin, alpha=alpha, dropout=dropout)

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        The forward function.
        :param x: input tensor.
        :param seed:
        :param prngs_cuda:
        :return: scores, maps.
        """
        modalities = self.to_modalities(x)
        maps = self.to_maps(modalities)
        scores = self.wildcat(x=maps, seed=seed, prngs_cuda=prngs_cuda)

        return scores, maps

    def get_nbr_params(self):
        """
        Compute the number of parameters of the model.
        :return:
        """
        return sum([p.numel() for p in self.parameters()])

    def __str__(self):
        return "{}: WILDCAT.".format(self.name)
    
    def __repr__(self):
        return super(WildCatClassifierHead, self).__repr__()


if __name__ == "__main__":
    inst = WildCatClassifierHead(inplans=10, modalities=5, num_classes=2)
    print(repr(inst))
    print(inst)