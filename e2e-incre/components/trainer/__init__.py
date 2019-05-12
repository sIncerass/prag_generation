# Based on: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import logging
import os
import time

import numpy as np
import torch
from torch import optim

from components.constants import PAD_ID
from components.evaluator.e2e_evaluator import BaseEvaluator
from components.evaluator.evaluation import eval_output
from components.utils.serialization import save_model, save_scores
from components.utils.serialization import save_predictions_txt
from components.utils.timing import create_progress_bar, asMinutes
from components.utils.visualize import torch_summarize

logger = logging.getLogger('experiment')


class BaseTrainer(object):
    def __init__(self, config):
        self.config = config
        self.init_params()

    def init_params(self):
        self.n_epochs = self.config["n_epochs"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config['learning_rate']
        self.model_dir = self.config["model_dir"]
        self.evaluate_prediction = self.config["evaluate_prediction"]
        self.save_model = self.config["save_model_each_epoch"]

        self.use_cuda = torch.cuda.is_available()

        self.train_losses = []
        self.dev_losses = []

        if self.evaluate_prediction:
            self.nist_scores = []
            self.bleu_scores = []
            self.cider_scores = []
            self.rouge_scores = []
            self.meteor_scores = []

    def run_external_eval(self, ref_fn, pred_fn):

        """
        Run external evaluation script (provided by the E2E NLG org)

        :param ref_fn: reference filename
        :param pred_fn: prediction filename
        :return:
        """

        bleu, nist, meteor, rouge, cider = eval_output(ref_fn, pred_fn)
        self.bleu_scores.append(bleu)
        self.nist_scores.append(nist)
        self.cider_scores.append(cider)
        self.rouge_scores.append(rouge)
        self.meteor_scores.append(meteor)

        score_msg = 'BLEU=%0.5f NIST=%0.5f CIDEr=%0.5f ROUGE=%0.5f METEOR=%0.5f' \
                    % (bleu, nist, cider, rouge, meteor)

        logger.info(score_msg)

    def record_loss(self, train_loss, dev_loss):
        self.train_losses.append(train_loss)
        self.dev_losses.append(dev_loss)

        logger.info('tloss=%0.5f dloss=%0.5f' % (train_loss, dev_loss))

    def training_start(self, model, data):

        training_start_time = time.time()
        logger.info("Start training")

        # Print a model summary to make sure everything is ok with it
        model_summary = torch_summarize(model)
        logger.debug(model_summary)

        evaluator = BaseEvaluator(self.config)
        logger.debug("Preparing training data")

        train_batches = data.prepare_training_data(data.train, self.batch_size)
        dev_batches = data.prepare_training_data(data.dev, self.batch_size)

        id2word = data.vocab.id2tok
        dev_lexicalizations = data.lexicalizations['dev']
        dev_multi_ref_fn = '%s.multi-ref' % data.fnames['dev']

        self.set_optimizer(model, self.config['optimizer'])
        self.set_train_criterion(len(id2word), PAD_ID)
        #print(data.dev[0])
        #exit()
        # Moving the model to GPU, if available
        if self.use_cuda:
            model = model.cuda()

        for epoch_idx in range(1, self.n_epochs + 1):

            epoch_start = time.time()
            pred_fn = os.path.join(self.model_dir, 'predictions.epoch%d' % epoch_idx)

            train_loss = self.train_epoch(epoch_idx, model, train_batches)
            dev_loss = self.compute_val_loss(model, dev_batches)

            predicted_ids, attention_weights = evaluator.evaluate_model(model, data.dev[0], data.uni_mr['dev'], dis=False)
            #print(len(predicted_ids))
            predicted_tokens = evaluator.lexicalize_predictions(predicted_ids,
                                                                dev_lexicalizations,
                                                                id2word)

            save_predictions_txt(predicted_tokens, pred_fn)
            self.record_loss(train_loss, dev_loss)

            if self.evaluate_prediction:
                self.run_external_eval(dev_multi_ref_fn, pred_fn)

            if self.save_model:
                save_model(model, os.path.join(self.model_dir, 'weights.epoch%d' % epoch_idx))

            logger.info('Epoch %d/%d: time=%s' % (epoch_idx, self.n_epochs, asMinutes(time.time() - epoch_start)))

        self.plot_lcurve()

        if self.evaluate_prediction:
            score_fname = os.path.join(self.model_dir, 'scores.csv')
            scores = self.get_scores_to_save()
            save_scores(scores, self.score_file_header, score_fname)
            self.plot_training_results()

        logger.info('End training time=%s' % (asMinutes(time.time() - training_start_time)))

    def compute_val_loss(self, model, dev_batches):

        total_loss = 0
        running_losses = []
        num_dev_batches = len(dev_batches)
        bar = create_progress_bar('dev_loss')

        for batch_idx in bar(range(num_dev_batches)):
            loss_var = self.train_step(model, dev_batches[batch_idx])
            loss_data = loss_var.data[0]

            # Record loss
            running_losses = ([loss_data] + running_losses)[:20]
            bar.dynamic_messages['dev_loss'] = np.mean(running_losses)

            total_loss += loss_data

        total_loss_avg = total_loss / num_dev_batches
        return total_loss_avg

    def train_epoch(self, epoch_idx, model, train_batches):

        np.random.shuffle(train_batches)  # shuffling data
        running_losses = []
        epoch_losses = []

        num_train_batches = len(train_batches)
        bar = create_progress_bar('train_loss')

        for pair_idx in bar(range(num_train_batches)):
            self.optimizer.zero_grad()
            loss_var = self.train_step(model, train_batches[pair_idx])
            loss_data = loss_var.data[0]
            loss_var.backward()  # compute gradients
            self.optimizer.step()  # update weights

            running_losses = ([loss_data] + running_losses)[:20]
            bar.dynamic_messages['train_loss'] = np.mean(running_losses)
            epoch_losses.append(loss_data)

        epoch_loss_avg = np.mean(epoch_losses)

        return epoch_loss_avg

    def get_scores_to_save(self):
        scores = list(zip(self.bleu_scores,
                          self.nist_scores,
                          self.cider_scores,
                          self.rouge_scores,
                          self.meteor_scores,
                          self.train_losses,
                          self.dev_losses))

        return scores

    def train_step(self, *args, **kwargs):
        raise NotImplementedError()

    def calc_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def set_train_criterion(self, *args, **kwargs):
        raise NotImplementedError()

    def set_optimizer(self, model, opt_name):

        logger.debug("Setting %s as optimizer" % opt_name)

        if opt_name == "SGD":
            self.optimizer = optim.SGD(params=model.parameters(), lr=self.lr)

        elif opt_name == "Adam":
            self.optimizer = optim.Adam(params=model.parameters(), lr=self.lr)

        elif opt_name == 'RMSprop':
            self.optimizer = optim.RMSprop(params=model.parameters(), lr=self.lr)

        else:
            raise NotImplementedError()

    def plot_training_results(self, *args, **kwargs):
        raise NotImplementedError()

    def plot_lcurve(self, *args, **kwargs):
        raise NotImplementedError()

    def get_plot_names(self):
        raise NotImplementedError()

    @property
    def score_file_header(self):
        HEADER = ['bleu', 'nist', 'cider', 'rouge', 'meteor', 'train_loss', 'dev_loss']
        return HEADER
