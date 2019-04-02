import logging

import components.constants as constants
from components.utils.io import check_file_exists

logger = logging.getLogger('experiment')

class VocabularyBase(object):
    """
    Common methods for all vocabulary classes.
    """

    def load_vocabulary(self, vocabulary_path):

        """
        Load vocabulary from file.
        """

        if check_file_exists([vocabulary_path]):
            logger.debug('Loading vocabulary from %s' % vocabulary_path)

            vocablist = []

            with open(vocabulary_path, 'r') as f:
                for line in f:
                    vocablist.append(line.strip())

            for idx, tok in enumerate(vocablist):
                self.id2tok[idx] = tok
                self.tok2id[tok] = idx

        else:
            raise ValueError('Vocabulary file not found: %s' % vocabulary_path)

    def process_raw_data(self, raw_data):

        vocablist = []

        for x in raw_data:
            for tok in x:
                if self.lower:
                    tok = tok.lower()

                if tok not in vocablist:
                    vocablist.append(tok)

        return vocablist

    def get_word(self, key, default=constants.UNK_ID):
        """
        Retrieve (numerical) id of a key, if present; default_id, otherwise.
        :param key: query token
        :param default: default numerical id
        :return: numerical id for the query token (int)
        """
        key = key.lower() if self.lower else key
        val = self.tok2id.get(key, default)
        return val

    def get_label(self, idx, default=None):
        """
        Retrieve a token corresponding to the query (numerical) id
        :param idx: (numerical) query id
        :param default: default token to return, if the idx is not in the vocabulary
        :return:
        """
        return self.id2tok[idx]

    def __len__(self):
        return self.size

    @property
    def size(self):
        return len(self.id2tok)


class VocabularyOneSide(VocabularyBase):
    def __init__(self, vocab_path, data_raw=None, lower=True):
        """
        Initialize a vocabulary class. Either you specify a vocabulary path to load
        the vocabulary from a file, or you provide training data to create one.
        :param vocab_path: path to a saved vocabulary
        :param data_raw: training data
        """
        self.lower = lower
        self.id2tok = {}
        self.tok2id = {}

        if not check_file_exists(vocab_path):
            assert data_raw is not None, "You need to process train data ** before ** creating a vocabulary!"
            self.create_vocabulary(raw_data=data_raw, vocab_path=vocab_path)

        else:
            # Load a saved vocabulary
            self.load_vocabulary(vocab_path)

    def create_vocabulary(self, raw_data, vocab_path):
        """
        A simple way to create a vocabulary.

        :param raw_data: data in the form of a list
        :param vocab_path: filename to save the vocabulary
        :return:
        """
        logger.info('Creating vocabulary')

        assert type(raw_data) == list

        vocablist = constants.START_VOCAB
        vocablist.extend(self.process_raw_data(raw_data))
        vocablist = list(set(vocablist))

        # saving the vocabulary
        with open(vocab_path, 'w') as vocab_file:
            for w in vocablist:
                vocab_file.write('%s\n' % w)

        for idx, tok in enumerate(vocablist):
            self.id2tok[idx] = tok
            self.tok2id[tok] = idx

        logger.info('Created vocabulary of size %d' % self.size)


class VocabularyShared(VocabularyBase):
    def __init__(self, vocab_path, data_raw_src=None, data_raw_tgt=None, lower=True):
        """
        Initialize a vocabulary class. Either you specify a vocabulary path to load
        the vocabulary from a file, or you provide training data to create one.
        :param vocab_path: path to a saved vocabulary
        :param data_raw_src: training data, source side
        :param data_raw_tgt: training data, target side
        """
        self.lower = lower
        self.id2tok = {}
        self.tok2id = {}

        if not check_file_exists(vocab_path):
            assert (data_raw_src is not None) and (data_raw_tgt is not None), \
                "You need to process train data ** before ** creating a vocabulary!"
            self.create_vocabulary(raw_data_src=data_raw_src,
                                   raw_data_tgt=data_raw_tgt,
                                   vocab_path=vocab_path)

        else:
            # Load a saved vocabulary
            self.load_vocabulary(vocab_path)

    def create_vocabulary(self, raw_data_src, raw_data_tgt, vocab_path):
        """
        A simple way to create a vocabulary.

        :param raw_data: data in the form of a list
        :param vocab_path: filename to save the vocabulary
        :return:
        """
        logger.info('Creating vocabulary')

        assert (type(raw_data_src) == type(raw_data_tgt) == list)

        vocablist = constants.START_VOCAB
        vocablist.extend(self.process_raw_data(raw_data_src))
        vocablist.extend(self.process_raw_data(raw_data_tgt))

        # saving the vocabulary
        with open(vocab_path, 'w') as vocab_file:
            for w in vocablist:
                vocab_file.write('%s\n' % w)

        for idx, tok in enumerate(vocablist):
            self.id2tok[idx] = tok
            self.tok2id[tok] = idx

        logger.info('Created vocabulary of size %d' % self.size)
