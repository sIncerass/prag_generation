import csv
import logging
import re

from components.constants import MR_FIELDS, NAME_TOKEN, NEAR_TOKEN
from components.data.vocabulary import VocabularyShared

logger = logging.getLogger('experiment')

# Regular expressions used to tokenize strings.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_FRAC_NUM_PAT = re.compile(r'(.?)(\d+\.\d+)')
_ZERO_PAT = re.compile(r'.?\d+\.00')


class BaseDataClass(object):
    """
    Base class for objects dealing with data pprerocessing
    and preparing data for training and evaluating the models.
    """

    def __init__(self, config):
        self.config = config or dict()

        # Following the tradition in MT, specifying both src and tgt (but will use either one)
        self.max_src_len = self.config["max_src_len"]
        self.max_tgt_len = self.config["max_tgt_len"]

        # constant (8)
        self.num_mr_attr = len(MR_FIELDS)
        self.vocab = None
        self.uni_mr = {'train': None, 'dev': None, 'test': None}
        self.lexicalizations = {'train': None, 'dev': None, 'test': None}
        self.fnames = {}

    def setup(self):
        logger.info("Setting up the data")

        train_x_raw = train_y_raw = None
        dev_x_raw = dev_y_raw = None
        test_x_raw = test_y_raw = None

        train_data_fname = self.config.get("train_data", None)
        dev_data_fname = self.config.get("dev_data", None)
        test_data_fname = self.config.get("test_data", None)
        vocab_path = "%s.vocab" % (train_data_fname)

        # Read train data, if the train data filename is specified in the config file
        if train_data_fname is not None:
            logger.debug("Reading train data")
            train_x_raw, train_y_raw, train_lex, train_mr = self.read_csv_train(train_data_fname, group_ref=True)
            self.lexicalizations['train'] = train_lex
            self.uni_mr['train'] = train_mr
            # Read dev data, if the dev data filename is specified in the config file
            # Note that training only makes sense if we have both train and dev data
            if dev_data_fname is None:
                raise Exception("No dev data in the config file!")

            logger.debug("Reading dev data")
            dev_x_raw, dev_y_raw, dev_lex, dev_mr = self.read_csv_train(dev_data_fname, group_ref=True)
            self.lexicalizations['dev'] = dev_lex
            self.uni_mr['dev'] = dev_mr

        # Read test data, if test data filename is specified in the config file
        if test_data_fname is not None:
            logger.debug("Reading test data")
            test_x_raw, test_y_raw, test_lex, test_mr = self.read_csv_train(test_data_fname)#self.read_csv_test(test_data_fname)
            self.lexicalizations['test'] = test_lex
            self.uni_mr['test'] = test_mr
        # Setup vocabulary
        self.setup_vocab(vocab_path, train_x_raw, train_y_raw)

        if train_x_raw is not None:
            self.train = self.data_to_token_ids_train(train_x_raw, train_y_raw)
            self.fnames['train'] = train_data_fname

        if dev_x_raw is not None:
            self.dev = self.data_to_token_ids_train(dev_x_raw, dev_y_raw)
            self.fnames['dev'] = dev_data_fname

        if test_x_raw is not None:
            self.test = self.data_to_token_ids_test(test_x_raw)
            self.fnames['test'] = test_data_fname

    def read_csv_train(self, fname, group_ref=False):
        """
        Read the CSV file containing training data.

        :param fname:
        :param group_ref: group multiple references, if possible (dev set)
        :return: 3 lists:
            - MR instances
            - corresponding textual descriptions
            - lexicalizations of ['name', 'near'] for each instance
        """

        raw_data_x = []
        raw_data_y = []
        lexicalizations = []
        uni_mrs = []
        orig = []
        with open(fname, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            header = next(reader)

            # Files should have headers
            assert header == ['mr', 'ref'], 'The file does not contain a header!'

            first_row = next(reader)
            curr_mr = first_row[0]
            curr_snt = first_row[1]
            orig.append((curr_mr, curr_snt))

            curr_src, curr_lex = self.process_e2e_mr(curr_mr)
            curr_text = self.tokenize(curr_snt, curr_lex)

            # add raw data instance
            raw_data_x.append(curr_src)
            raw_data_y.append(curr_text)
            lexicalizations.append(curr_lex)
            uni_mrs.append(curr_mr)
            for row in list(reader):
                mr = row[0]
                text = row[1]
                orig.append((mr, text))

                this_src, this_lex = self.process_e2e_mr(mr)
                this_text = self.tokenize(text, this_lex)

                # add raw data instance
                raw_data_x.append(this_src)
                raw_data_y.append(this_text)
                uni_mrs.append(mr)
                if mr == curr_mr:
                    continue
                else:
                    lexicalizations.append(this_lex)
                    curr_mr = mr
                """
                if this_src == curr_src:
                    continue

                else:
                    lexicalizations.append(this_lex)
                    curr_src = this_src
                """
        if group_ref:
            gen_multi_ref_dev(orig, fname='%s.multi-ref' % fname)
        print(len(raw_data_x), len(raw_data_y), len(lexicalizations), len(uni_mrs))
        return raw_data_x, raw_data_y, lexicalizations, uni_mrs

    def read_csv_test(self, fname):
        raw_data_x = []
        lexicalizations = []
        uni_mrs = [] 
        with open(fname, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            header = next(reader)
            for row in list(reader):
                mr = row[0]
                this_src, this_lex = self.process_e2e_mr(mr)

                # add raw data instance
                raw_data_x.append(this_src)
                lexicalizations.append(this_lex)
                uni_mrs.append(mr)
        return raw_data_x, None, lexicalizations, uni_mrs

    def tokenize_normalize(self, s, lex_list=None):
        """
        Experimenting with various ways to normalize the data.
        As it turned out, normalization did not improve the results.
        Reason: all data, including dev set follows the same noisy distribution.
        So, normalization results in enlarging the distance between data distributions for train and dev sets.
        :param s: target string
        :param lex_list: list containing lexicalizations for the string s
        :return:
        """
        words = []

        # Delexicalize target side
        if lex_list:
            for l, t in zip(lex_list, (NAME_TOKEN, NEAR_TOKEN)):
                if l:
                    s = s.replace(l, t)
        s_r_toks, s_toks = [], s.strip().split()
        for tok in s_toks:
            if NEAR_TOKEN in tok:
                tok = tok.split(NEAR_TOKEN)
                tok = [NEAR_TOKEN if x == '' else x for x in tok]
                s_r_toks.extend(tok)
            elif NAME_TOKEN in tok:
                tok = tok.split(NAME_TOKEN)
                tok = [NAME_TOKEN if x == '' else x for x in tok]
                s_r_toks.extend(tok)
            else:
                s_r_toks.append(tok)
        print(s_r_toks, s_tpks)
        # Process target side text
        for fragment in s_r_toks:#s.strip().split():

            # normalizing the price
            match = _FRAC_NUM_PAT.match(fragment)
            if match:
                fragment_tokens = []
                pound = match.group(1)
                price = match.group(2)
                price = re.sub(r'.00', '', price)  # stripping off trailing zeros from the price
                if not pound.isdigit():
                    fragment_tokens.append(pound)
                fragment_tokens.append(price)

            match = _ZERO_PAT.match(fragment)
            if match:

                fragment_tokens = [re.sub('.00', '', fragment)]

            else:
                fragment_tokens = _WORD_SPLIT.split(fragment)
            words.extend(fragment_tokens)

        tokens = [w for w in words if w]

        return tokens

    def tokenize(self, s, lex_list=None):
        """
        A simple procedure to tokenize a string.

        :param s: string to be tokenized
        :param lex_list: list of lexicalizations
        :return: list of tokens from sting s
        """

        words = []

        # Delexicalize target side
        if lex_list:
            for l, t in zip(lex_list, (NAME_TOKEN, NEAR_TOKEN)):
                if l:
                    s = s.replace(l, t) 
        s_r_toks, s_toks = [], s.strip().split()
        for tok in s_toks:
            if NEAR_TOKEN != tok and NEAR_TOKEN in tok:
                tok = tok.split(NEAR_TOKEN)
                tok = [NEAR_TOKEN if x == '' else x for x in tok]
                s_r_toks.extend(tok)
            elif NAME_TOKEN != tok and NAME_TOKEN in tok:
                tok = tok.split(NAME_TOKEN)
                tok = [NAME_TOKEN if x == '' else x for x in tok]
                s_r_toks.extend(tok)
            else:
                s_r_toks.append(tok)
        #print(s_r_toks, s_toks)
        #exit()
        # Process target side text
        for fragment in s_r_toks:
        #for fragment in s.strip().split():
            fragment_tokens = _WORD_SPLIT.split(fragment)
            words.extend(fragment_tokens)

        tokens = [w for w in words if w]

        return tokens

    def setup_vocab(self, vocab_path, train_x_raw, train_y_raw):
        raise NotImplementedError()

    def process_e2e_mr(self, s):
        raise NotImplementedError()

    def data_to_token_ids_train(self, *args, **kwargs):
        raise NotImplementedError()

    def data_to_token_ids_test(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare_training_data(self, *args, **kwargs):
        raise NotImplementedError()


def gen_multi_ref_dev(dev_xy, fname):
    """
    A utility function for generating a mutli-reference file
    from the data provided by the E2E NLG organizers.

    :param dev_xy: a list of M tuples, where M is the number of data instances.
    Each tuple contains a src string and a tgt string.
    For example:
    ('name[Taste of Cambridge], eatType[restaurant], priceRange[£20-25], customer rating[3 out of 5]',
    'Taste of Cambridge is a restaurant with a customer rating of 3 out of 5 and and a price range of £20-£25')

    :param fname: the name of the target file, where you want to store multi-reference data
    :return:
    """

    logger.debug('Generaing a multi-ref file')

    multi_ref_src_fn = '%s.src' % fname
    with open(fname, 'w') as fout, open(multi_ref_src_fn, 'w') as fout_src:

        # Write the first sentence
        curr_mr, curr_txt = dev_xy[0]
        fout.write('%s\n' % curr_txt)
        fout_src.write('%s\n' % curr_mr)

        # Iterate over xy pairs
        for mr, txt in dev_xy[1:]:

            # If MR is the same, write the ref to the tgt file
            if mr == curr_mr:
                fout.write('%s\n' % txt)

            else:
                # Else, write a newline, then the new ref and update current MR
                fout.write('\n')
                fout.write('%s\n' % txt)
                curr_mr = mr

                # Write the new MR input to the src file
                fout_src.write('%s\n' % mr)

    logger.debug('Done!')
