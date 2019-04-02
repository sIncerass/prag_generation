import os
import torch
import logging
from datetime import datetime
import json
import csv

logger = logging.getLogger('experiment')


def save_config(config_dict, fname=None):
    """
    Save configuration dictionary (n json format).

    :param config_dict: configuration dictionary
    :param fname: name of the file to save the dictionary in
    :return:
    """

    with open(fname, mode='w', encoding='utf-8') as f:
        json.dump(config_dict, f)


def save_model(model, model_fn):
    """
    Serialize the trained model.

    :param model: instance of the ModelClass
    :param model_fn: name of the file where to store the model
    :return:
    """

    logger.info('Saving model to --> %s' % model_fn)
    torch.save(model.state_dict(), open(model_fn, 'wb'))


def save_predictions_json(predictions, fname):
    """
    Save predictions done by a trained model in json format.

    :param predictions: a list of strings, each corresponding to one predicted snt.
    :param fname: name of the file to save the predictions in
    :return:
    """

    with open(fname, mode='w', encoding='utf-8') as f:
        logger.info('Saving all predictions to a json file --> %s' % fname)
        json.dump(predictions, f)


def save_predictions_txt(predictions, fname):
    """
    Save predictions done by a trained model in txt format.
    :param predictions:
    :param fname:
    :return:
    """
    logger.info('Saving predictions to a txt file --> %s' % fname)

    with open(fname, mode='w', encoding='utf-8') as f:

        if type(predictions) == str:
            f.write(predictions)

        elif type(predictions) == list:
            for s in predictions:
                if type(s) == list:
                    s = ' '.join(s)

                f.write('%s\n' % s)

        else:
            raise NotImplementedError()


def save_scores(scores, header, fname):
    """
    Save the performance of the model as measured for each epoch by the E2E NLG Challenge scoring metrics.

    :param scores: a list of lists of scores:
            - bleu_scores
            - nist_scores
            - cider_scores
            - rouge_scores
            - meteor_scores
            - train_losses
            - dev_losses

    :param header: explains which scores are stored in which column of the CSV file
    :param fname:
    :return:
    """

    with open(fname, 'w') as csv_out:
        csv_writer = csv.writer(csv_out, delimiter=',')
        csv_writer.writerow(header)
        for epoch_scores in scores:
            csv_writer.writerow(epoch_scores)

    logger.info('Scores saved to --> %s' % fname)


def load_model(model, model_weights_fn):
    """
    Load serialized model.

    :param model: instance of the ModelClass.
    :param model_weights_fn: name of the file w/ the serialized model.
    :return:
    """

    logger.info('Loading the model <-- %s' % model_weights_fn)
    model.load_state_dict(torch.load(open(model_weights_fn, 'rb')))


def get_experiment_name(config_d):
    """
    Create a simple unique name for the experiment.
    Consists of model type, timestamp and specific hyper-parameter (hp) values.

    :param config_d: configuration dictionary
    :return: name of the current experiment (string)
    """

    # model type
    model_type = get_model_type(config_d)
    timestamp = get_timestamp()
    hp_name = get_hp_value_name(config_d)
    return '%s_%s_%s' % (model_type, hp_name, timestamp)


def get_model_type(config_d):
    """
    Generate part of the experiment name: model type to be trained.
    Model types correspond to non-qualified names of the python files with the code for the model.
    For example, if the code for our model is stored in "components/model/e2e_model_MLP.py",
    then model type is "e2e_model_MLP".

    :param config_d: configuration dictionary
    :return: model type (string)
    """
    mtype = config_d['model-module'].split('.')[-1]
    config_d['training_params']['modeltype'] = mtype
    return mtype


def get_timestamp():
    """
    Generate a timestep to be included as part of the model name.

    :return: current timestamp (string)
    """
    return '{:%Y-%b-%d_%H:%M:%S}'.format(datetime.now())


def get_hp_value_name(config_dict):
    """
    Generate a string which store hyper-parameter values as part of the model name.
    This is useful for hp-optimization, if you decide to perform one.

    :param config_dict: configuration dictionary retrieved from the .yaml file.
    :return: concatenated hp values (string)
    """

    seed = config_dict.get('random_seed', 1)
    embed = config_dict['model_params']['embedding_dim']
    hidden_size = config_dict['model_params']['encoder_params']['hidden_size']
    dropout = config_dict['model_params']['encoder_params']['dropout']
    batch_size = config_dict['training_params']['batch_size']
    lr = config_dict['training_params']['learning_rate']

    return 'seed%s-emb%s-hid%s-drop%s-bs%s-lr%s' % (seed, embed, hidden_size, dropout, batch_size, lr)


def make_model_dir(config):
    """
    Create a directory to contain various files for the current experiment.
    :param config: config dictionary
    :return: name of the directory
    """

    # Experiment directory name differs depending on the mode
    mode = config['mode']

    if mode == 'predict':
        model_fn = config["model_fn"]
        model_dirname = os.path.split(model_fn)[0]  # dir which holds the serialzed (trained) model

    elif mode == 'train':
        # need to have only one directory for all experiments
        # if it does not exist already, it will be created
        all_experiments_dir = os.path.abspath(config["experiments_dir"])

        # name of the model to be trained
        model_name = get_experiment_name(config)

        # here we specify the unique name for this experiment
        # by combining the experiment dir and a unique model name
        model_dirname = os.path.join(all_experiments_dir, model_name)

    else:
        raise NotImplementedError()

    # create the experiment directory, if it does not exist
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

    # store the directory name in the config file
    config['training_params']['model_dir'] = model_dirname

    return model_dirname


def test_save_scores():
    scores = [[(1, 2), 3], [(1, 2), 4], [(1, 2), 5]]

    HEADER = ['loss', 'cider']

    with open('todelete.csv', 'w') as csv_out:
        csv_writer = csv.writer(csv_out, delimiter=',')
        csv_writer.writerow(HEADER)
        for epoch_scores in scores:
            csv_writer.writerow(epoch_scores)


if __name__ == '__main__':
    test_save_scores()
