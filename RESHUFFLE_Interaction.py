import torch
from torch import FloatTensor
from pykeen.nn.modules import FunctionalInteraction
from Utils import relation_multiplication


def reshuffle_interaction(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the RESHUFFLE interaction function.

    :param h: shape: (`*batch_dims`, k, l)
        The head representations.
    :param r: shape: (`*batch_dims`, l, l)
        The relation representations.
    :param t: shape: (`*batch_dims`, k, l)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """

    res = -torch.norm(torch.relu(relation_multiplication(r, h) - t), dim=(-2, -1), p=2)
    return res


class RESHUFFLE_Interaction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """
    A module wrapper for the stateless RESHUFFLE interaction function.
    """

    entity_shape = ('kl',)
    relation_shape = ('dl',)  # dl is here necessary to constrain each row of the relation matrices to have at most one value that is 1
                              # -> this is accomplished with a row-wise softmax on the relation matrices and by subsequently dropping a column

    func = reshuffle_interaction
