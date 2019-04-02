import logging
import os
import sys


def set_logger(stdout_level=logging.INFO, log_fn=None):
    """
    Set python logger for this experiment.
    Based on:

        https://stackoverflow.com/questions/25187083/python-logging-to-multiple-handlers-at-different-log-levels

    :param stdout_level:
    :param log_fn:
    :return:
    """

    # create formatters
    simple_formatter = logging.Formatter("%(name)s:%(levelname)s: %(message)s")
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # get a top-level "mypackage" logger,
    # set its log level to DEBUG,
    # BUT PREVENT IT from propagating messages to the root logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.DEBUG)
    logger.propagate = 0

    # create a console handler
    # and set its log level to the command-line option
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, stdout_level))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    if log_fn:
        # create a file handler
        # and set its log level to DEBUG
        log_fn = os.path.abspath(log_fn)

        file_handler = logging.FileHandler(log_fn)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger
