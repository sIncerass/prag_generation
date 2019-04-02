import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import json
from torch.nn.modules.module import _addindent
import torch

# needed on machines w/o DISPLAY var set
plt.switch_backend('agg')

logger = logging.getLogger('experiment')

PLOT_ARGS = ['xr-',
             'xg-',
             'ob-',
             'or--',
             'og--',
             'ob--',
             'om-']


def plot_train_progress(scores, img_title, save_path, show, names=None):
    """
    A plotting function using the array of loss values saved while training.
    :param train_losses, dev_losses: losses saved during training
    :return:
    """

    nrows, ncols = 2, 3
    dx, dy = 2, 1
    num_iter = len(scores[0])
    xs = np.arange(start=1, stop=num_iter + 1, step=1)
    figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(img_title)

    for sc, ax, name in zip(scores, axes.flat, names):

        # Set label for the X axis
        ax.set_xlabel('EpochN', fontsize=12)

        if type(name) in [list, tuple]:  # this should happen with loss plotting only
            # It means that scores are represented as an MxN Numpy array
            num_curves = sc.shape[1]
            for idx in range(num_curves):
                ax.plot(xs, sc[:, idx])

            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.legend(name)  # name is a list -> need to create a legend for this subplot
            ax.set_ylabel('Loss', fontsize=12)

        else:
            ax.plot(xs, sc)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_ylabel(name, fontsize=12)

    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    pad = 0.05  # Padding around the edge of the figure
    xpad, ypad = dx * pad, dy * pad
    fig.tight_layout(pad=2, h_pad=xpad, w_pad=xpad)

    if save_path is not None:
        logger.debug("Saving the learning curve plot --> %s" % save_path)
        fig.savefig(save_path)

    if show:
        plt.show()


def plot_lcurve(train_loss, dev_loss, img_title, save_path, show):
    num_iter = len(train_loss)
    xs = np.arange(start=1, stop=num_iter + 1, step=1)
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title(img_title)
    plt.legend(["TrainLoss", "DevLoss"])
    plt.legend(loc='best', fancybox=True, framealpha=0.5)

    if save_path is not None:
        logger.debug("Saving the learning curve plot --> %s" % save_path)
        plt.savefig(save_path)

    if show:
        plt.show()


def print_predictions(src_json_fname):
    with open(src_json_fname, mode='r', encoding='utf-8') as fin:
        predicted_ids_all_epochs = json.load(fin)

        for epoch_id, epoch_snts in enumerate(predicted_ids_all_epochs):

            with open('%s-%d.txt' % (src_json_fname, epoch_id), mode='w', encoding='utf-8') as fout:
                for snt_tokens in epoch_snts:
                    fout.write('%s\n' % ' '.join(snt_tokens))

    print('Done writing predictions')


def torch_summarize(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights.
    Taken from:
    https://stackoverflow.com/questions/42480111/model-summary-in-pytorch#42616812
    """

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr
