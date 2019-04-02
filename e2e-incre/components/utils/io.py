import os

from components.utils.config import logger


def check_file_exists(fname):
    if not os.path.exists(os.path.abspath(fname)):
        logger.warning("%s does not exist!" % (fname))
        return False


def check_files_exist(args):
    for arg in args:
        check_file_exists(arg)

    return True
