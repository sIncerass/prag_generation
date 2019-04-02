import copy

import numpy as np
import torch
from torch.autograd import Variable
import functools
from components.constants import BOS_ID, EOS_ID, PAD_ID


def truncate_pad_sentence(snt_ids, max_len, add_start=True, add_end=True):
    """
    Given an iterable
    :param snt_ids:
    :param max_len:
    :param add_start:
    :param add_end:
    :return:
    """
    x_trunc = truncate_sentence(snt_ids, max_len, add_start, add_end)
    x_trunc_pad = pad_snt(x_trunc, max_len)

    return x_trunc_pad


def truncate_sentence(snt_ids, max_len, add_start=True, add_end=True):
    """
    Given a list of token ids and maxium sequence length,
    take only the first "max_len" items.

    :param snt_ids: sequence of elements (token ids)
    :param max_len: cut-off value for truncation
    :param add_start: add "beginning-of-sentence" symbol (BOS) (bool)
    :param add_end: add "end-of-sentence" symbol (EOS) (bool)
    :return:
    """
    if add_start:
        snt_ids = [BOS_ID] + snt_ids

    if add_end:
        snt_ids = snt_ids + [EOS_ID]

    return snt_ids[:max_len]


def pad_snt(snt_ids_trunc, max_len):
    """
    Given a list of token ids and maxium sequence length,
    pad the sentence, if necessary, so that it contains exactly "max_len" items.

    :param snt_ids_trunc: possibly truncated sequence of items to be padded
    :param max_len: cut-off value for padding
    :return:
    """

    snt_ids_trunc_pad = snt_ids_trunc + [PAD_ID] * (max_len - len(snt_ids_trunc))

    return snt_ids_trunc_pad


def generator_wrapper(iterable):
    """
    A utility function wrapping an iterable into a generator.

    :param iterable:
    :return:
    """

    num_items = len(iterable)
    for idx in range(num_items):
        yield iterable[idx]


# ------------------- Cudifying tensors

# a bool value signifying that a GPU is available on the system
use_cuda = torch.cuda.is_available()


def cuda_if_gpu(T):
    """
    Move tensor to GPU, if one is available.

    :param T:
    :return:
    """

    return T.cuda() if use_cuda else T


def cudify(fn):
    """
    A simple decorator for wrapping functions that return tensors
    to move them to a GPU, if one is available.

    :param fn: function to be wrapped
    :return:
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return cuda_if_gpu(result)

    return wrapper


@cudify
def ids2var(snt_ids, *dims, addEOS=True):
    """
    Convert a sequence of ids to a matrix of size specified by the dims.

    :param snt_ids: s
    :param addEOS:
    :return: A matrix of shape: (*dims)
    """

    snt_ids_copy = copy.deepcopy(snt_ids)
    if addEOS:
        snt_ids_copy.append(EOS_ID)

    result = Variable(torch.LongTensor(snt_ids_copy).view(dims))

    return result


@cudify
def cat2onehot_var(snt_ids, vocab_size, batch_size):
    """
    Convert a sequence of categorical values to one-hot representation
    Based on: https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy#38592416
    """

    targets = np.array([snt_ids]).reshape(-1)
    one_hot_targets = np.eye(vocab_size)[targets]
    result = Variable(torch.FloatTensor(one_hot_targets).view(-1, batch_size, vocab_size))  #

    return result


def test_ids2var():
    snt_ids = [1, 1, 1, 1, 1]
    snt_ids_var = ids2var(snt_ids, 1, 2, 3, addEOS=True)
    snt_ids_var_shape = snt_ids_var.data.size()
    assert snt_ids_var_shape == torch.Size([1, 2, 3])
    print('Test (ids2var): passed')


if __name__ == "__main__":
    test_ids2var()
