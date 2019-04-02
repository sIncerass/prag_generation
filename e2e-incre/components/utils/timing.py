import math
import progressbar
import time


# Taken from:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

def timeSince(since, percent):
    """
    A helper function to print time elapsed and
    estimated time remaining given the current time and progress
    :param since:
    :param percent:
    :return:
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    """
    A helper function to convert elapsed time to minutes.
    :param s:
    :return:
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def create_progress_bar(dynamic_msg=None):
    # Taken from Andreas Rueckle.
    # usage:
    #   bar = _create_progress_bar('loss')
    #   L = []
    #   for i in bar(iterable):
    #       ...
    #       L.append(...)
    #
    #   bar.dynamic_messages['loss'] = np.mean(L)
    widgets = [
        ' [batch ', progressbar.SimpleProgress(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') '
    ]
    if dynamic_msg is not None:
        widgets.append(progressbar.DynamicMessage(dynamic_msg))
    return progressbar.ProgressBar(widgets=widgets)
