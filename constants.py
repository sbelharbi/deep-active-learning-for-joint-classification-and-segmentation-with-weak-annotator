# baseline
AL_WSL = "WSL"  # using only global annotation over the enter dataset.
# similar to AL_FULL_SUP but no segmentation training is done. The predicted
# mask is estimated from the CAMs.

# type of active learning
AL_RANDOM = 'Random'  # random selection
AL_LP = "Label_prop"  # our method. (label-propagation)
AL_FULL_SUP = "Full_sup"  # no active learning. use entire dataset with full
# supervision
AL_ENTROPY = "Entropy"  # uncertainty based on entropy of the classification
# scores.
AL_MCDROPOUT = "MC_Dropout"  # uncertainty based on MC dropout.

al_types = [AL_WSL, AL_RANDOM, AL_LP, AL_FULL_SUP, AL_ENTROPY, AL_MCDROPOUT]

# loss
CE = 'CE'
KL = 'KL'

# total segmentation loss: seg + cl.
HYBRIDLOSS = "HybridLoss"

losses = [CE, KL, HYBRIDLOSS]

# how to cluster samples in our method for selection.
CLUSTER_DENSITY_DIVERSITY = 'Density_and_diversity'
CLUSTER_DENSITY_LABELLING = 'Density_and_labelling'
CLUSTER_DENSITY_DIVERSITY_LABELLING = "Density,_diversity,_and_labelling"
# clustering that is based on standard active learning selection criteria.

CLUSTER_RANDOM = 'Random'  # random sampling.
CLUSTER_ENTROPY = 'Entropy'  # entropy sampling.
ours_clustering = [CLUSTER_DENSITY_DIVERSITY,
                   CLUSTER_DENSITY_LABELLING,
                   CLUSTER_DENSITY_DIVERSITY_LABELLING,
                   CLUSTER_RANDOM,
                   CLUSTER_ENTROPY]

# models
LENET5 = "lenet5"  # lenet5
SOTASSL = "sota_ssl"  # sota_ssl
HYBRIDMODEL = 'hybrid_model'  # for SEG task.

nets = [LENET5, SOTASSL, HYBRIDMODEL]

# datasets
# CL
CIFAR_10 = "cifar-10"
CIFAR_100 = "cifar-100"
SVHN = "svhn"
MNIST = "mnist"

# SEG
GLAS = "glas"
CUB = "Caltech-UCSD-Birds-200-2011"
OXF = "Oxford-flowers-102"
CAM16 = "camelyon16"

datasets = [CIFAR_10, CIFAR_100, SVHN, MNIST, GLAS]
CL_DATASETS = [CIFAR_10, CIFAR_100, SVHN, MNIST]
SEG_DATASETS = [GLAS, CUB, OXF]


# task
CL = 'CLASSIFICATION'
SEG = 'SEGMENTATION'
tasks = [CL, SEG]

# subtasks
SUBCL = CL
SUBSEG = SEG
SUBCLSEG = "Classification_Segmentation"
subtasks = [SUBCL, SUBSEG, SUBCLSEG]
# ==============================================================================
# Types of attention
NONE = 'NONE'
LEARNABLE = 'LEARNABLE'
STOCHASTICXXX = 'STOCHASTICXXX'

attention_types = [NONE, LEARNABLE, STOCHASTICXXX]
attentionz = [LEARNABLE, STOCHASTICXXX]

# Types of similarity measure between scores
JSD = "JSD"  # "Jensen-Shannon divergence"
MSE = "MSE"  # "Mean-squarred error"

sim_scores = [JSD, MSE]


# Tags for samples
L = 0  # Labeled samples
U = 1  # Unlabeled sample
PL = 2  # unlabeled sample that has been Pseudo-Labeled.

samples_tags = [L, U, PL]  # list of possible sample tags.

# indicator on how to find the best when looking to labele unlabeled samples.
LOW = "low"
HIGH = "high"

best = [LOW, HIGH]  # list of possible choices for the best criterion.

# Colours
COLOR_WHITE = "white"
COLOR_BLACK = "black"

# backbones.

RESNET18 = "resnet18"
RESNET34 = "resnet34"
RESNET50 = 'resnet50'
RESNET101 = 'resnet101'
RESNET152 = 'resnet152'
RESNEXT50_32X4D = 'resnext50_32x4d'
RESNEXT101_32X8D = 'resnext101_32x8d'
WIDE_RESNET50_2 = 'wide_resnet50_2'
WIDE_RESNET101_2 = 'wide_resnet101_2'

backbones = [RESNET18,
             RESNET34,
             RESNET50,
             RESNET101,
             RESNET152,
             RESNEXT50_32X4D,
             RESNEXT101_32X8D,
             WIDE_RESNET50_2,
             WIDE_RESNET101_2
             ]
resnet_backbones = [RESNET18,
                    RESNET34,
                    RESNET50,
                    RESNET101,
                    RESNET152,
                    RESNEXT50_32X4D,
                    RESNEXT101_32X8D,
                    WIDE_RESNET50_2,
                    WIDE_RESNET101_2
                    ]

# segmentation losses
BinSoftInvDiceLoss = 'BinSoftInvDiceLoss'
BinCrossEntropySegmLoss = 'BinCrossEntropySegmLoss'
BCEAndSoftDiceLoss = 'BCEAndSoftDiceLoss'
BinL1SegmLoss = 'BinL1SegmLoss'
BinIOUSegmLoss = 'BinIOUSegmLoss'
BinL1SegAndIOUSegmLoss = 'BinL1SegAndIOUSegmLoss'

seglosses = [
    BinCrossEntropySegmLoss, BinSoftInvDiceLoss, BCEAndSoftDiceLoss,
    BinL1SegmLoss, BinIOUSegmLoss, BinL1SegAndIOUSegmLoss
]

# scale decay

ConstantWeight = 'ConstantWeight'
LinearAnnealedWeight = 'LinearAnnealedWeight'
ExponentialDecayWeight = 'ExponentialDecayWeight'

scale_decay = [ConstantWeight, LinearAnnealedWeight,ExponentialDecayWeight]

