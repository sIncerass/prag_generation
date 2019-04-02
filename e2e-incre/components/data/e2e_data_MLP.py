from __future__ import unicode_literals, print_function, division

import numpy as np
import copy
import logging

from components.data import BaseDataClass
from components.data.vocabulary import VocabularyShared
from components.data.common import EOS_ID, PAD_ID, pad_snt
from components.constants import NAME_TOKEN, NEAR_TOKEN, MR_KEYMAP

logger = logging.getLogger('experiment')
MR_KEY_NUM = len(MR_KEYMAP)


class E2EMLPData(BaseDataClass):
    def process_e2e_mr(self, mr_string):
        """
        Processing E2E NLG Challenge meaning representation
        Represent each MR as a list of 8 attributes, specified in MR_KEYMAP.

        :param mr_string:
        :return:
        """
        items = mr_string.split(", ")
        mr_data = [PAD_ID] * MR_KEY_NUM
        lex = [None, None]  # holds lexicalized variants of NAME and NEAR

        for idx, item in enumerate(items):
            key, raw_val = item.split("[")
            key_idx = MR_KEYMAP[key]

            # Delexicalization of the 'name' field
            if key == 'name':
                mr_val = NAME_TOKEN
                lex[0] = raw_val[:-1]

            # Delexicalization of the 'near' field
            elif key == 'near':
                mr_val = NEAR_TOKEN
                lex[1] = raw_val[:-1]

            else:
                mr_val = raw_val[:-1]

            mr_data[key_idx] = mr_val

        return mr_data, lex

    def data_to_token_ids_train(self, raw_x, raw_y):
        """
        Convert raw lists of tokens to numerical representations.

        :param raw_x: a list of N instances, each being a list of MR values; N = size(dataset)
        :param raw_y: a list of N instances, each being a list of tokens,
        comprising textual description of the restaurant x
        :return:
        """

        assert self.max_src_len is not None
        assert self.max_tgt_len is not None

        data_split_x = []
        data_split_y = []
        skipped_cnt = 0

        for idx, x in enumerate(raw_x):
            src_ids = [self.vocab.get_word(tok) for tok in x]
            src_len = len(src_ids)

            y = raw_y[idx]
            tgt_ids = [self.vocab.get_word(tok) for tok in y]
            tgt_len = len(tgt_ids)

            # Truncating long sentences
            if src_len > self.max_src_len or tgt_len >= self.max_tgt_len:
                logger.info("Skipping long snt: %d (src) / %d (tgt)" % (src_len, tgt_len))
                skipped_cnt += 1
                continue

            data_split_x.append(src_ids)
            data_split_y.append(tgt_ids)

        logger.debug("Skipped %d long sentences" % skipped_cnt)
        return (data_split_x, data_split_y)

    def data_to_token_ids_test(self, raw_x):
        assert self.max_src_len is not None
        data_split_x = []
        for idx, x in enumerate(raw_x):
            src_ids = [self.vocab.get_word(tok) for tok in x]
            src_size = len(src_ids)

            # Truncating long sentences
            if src_size > self.max_src_len:
                logger.debug("Truncating long snt: %d" % idx)
                continue

            data_split_x.append(src_ids)

        return (data_split_x, None)

    def index_data(self, data_size, mode="no_shuffling"):
        """
        Aux function for indexing instances in the dataset.

        :param data_size:
        :param mode:
        :return:
        """

        if mode == "random":
            indices = np.random.choice(np.arange(data_size), data_size, replace=False)

        elif mode == "no_shuffling":
            indices = np.arange(data_size)

        else:
            raise NotImplementedError()

        return indices

    def prepare_training_data(self, xy_ids, batch_size):

        """
        Cut the dataset into batches.
        :param xy_ids: a tuple of 2 lists:
            - a list of MR instances, each itself being a list of numerical ids
            - a list ot tokenized texts (same format)
        :param batch_size:
        :return:
        """

        # sort according to the lengths -> xy pairs
        sorted_data = sorted(zip(*xy_ids), key=lambda p: len(p[0]), reverse=True)
        data_size = len(sorted_data)
        num_batches = data_size // batch_size
        data_indices = self.index_data(data_size, mode='no_shuffling')  # TODO: use shuffling
        batch_pairs = []

        for bi in range(num_batches):
            batch_x = []
            batch_y = []
            curr_batch_indices = data_indices[bi * batch_size: (bi + 1) * batch_size]

            for idx in curr_batch_indices:
                x_ids, y_ids = sorted_data[idx]

                x_ids_copy = copy.deepcopy(x_ids)
                x_ids_copy.append(EOS_ID)
                batch_x.append(x_ids_copy)

                y_ids_copy = copy.deepcopy(y_ids)
                y_ids_copy.append(EOS_ID)
                batch_y.append(y_ids_copy)

            batch_x_lens = [len(s) for s in batch_x]
            batch_y_lens = [len(s) for s in batch_y]

            max_src_len = max(batch_x_lens)
            max_tgt_len = max(batch_y_lens)

            batch_x_padded = [pad_snt(x, max_src_len) for x in batch_x]
            batch_y_padded = [pad_snt(y, max_tgt_len) for y in batch_y]

            batch_pairs.append((batch_x_padded, batch_y_padded))

        return batch_pairs

    def setup_vocab(self, vocab_path, train_x_raw, train_y_raw):
        self.vocab = VocabularyShared(vocab_path, train_x_raw, train_y_raw, lower=False)  # TODO: lower!


component = E2EMLPData
