import importlib
import os
import sys

from components.utils.config import load_config, fix_seed
from components.utils.log import set_logger
from components.utils.serialization import load_model, make_model_dir
from components.utils.serialization import save_config, save_predictions_txt


def run(config_dict):

    # Fetch all relevant modules.
    data_module = config_dict['data-module']
    model_module = config_dict['model-module']
    training_module = config_dict['training-module']
    evaluation_module = config_dict.get('evaluation-module', None)
    mode = config_dict['mode']

    # Load the modules
    DataClass = importlib.import_module(data_module).component
    ModelClass = importlib.import_module(model_module).component
    TrainingClass = importlib.import_module(training_module).component
    EvaluationClass = importlib.import_module(evaluation_module).component if evaluation_module else None

    model_dirname = make_model_dir(config_dict)
    logger = set_logger(config_dict["log_level"], os.path.join(model_dirname, "log.txt"))

    # Setup the data
    data = DataClass(config_dict["data_params"])
    data.setup()

    # Setup the model
    fix_seed(config_d['random_seed'])  # fix seed generators
    model = ModelClass(config_dict["model_params"])
    print("build model done")
    model.setup(data)  # there are some data-specific params => pass data as arg
    print("setup data done")
    if mode == "train":
        training_params = config_dict['training_params']
        trainer = TrainingClass(training_params)
        trainer.training_start(model, data)
        save_config(config_dict, os.path.join(model_dirname, 'config.json'))

    elif mode == "predict":
        assert evaluation_module is not None, "No evaluation module -- check config file!"
        evaluator = EvaluationClass(config_dict)
        model_fname = config_dict["model_fn"]
        load_model(model, model_fname)
        id2word = data.vocab.id2tok

        if 'dev' in data.fnames:
            logger.info("Predicting on dev data")
            predicted_ids, attention_weights = evaluator.evaluate_model(model, data.dev[0])
            data_lexicalizations = data.lexicalizations['dev']
            predicted_snts = evaluator.lexicalize_predictions(predicted_ids,
                                                              data_lexicalizations,
                                                              id2word)

            save_predictions_txt(predicted_snts, '%s.devset.predictions.txt' % model_fname)
            
        if 'test' in data.fnames:
            logger.info("Predicting on test data")
            predicted_ids, attention_weights = evaluator.evaluate_model(model, data.test[0])
            data_lexicalizations = data.lexicalizations['test']
            predicted_snts = evaluator.lexicalize_predictions(predicted_ids,
                                                              data_lexicalizations,
                                                              id2word)

            save_predictions_txt(predicted_snts, '%s.testset.predictions.txt' % model_fname)

    else:
        logger.warning("Check the 'mode' field in the config file: %s" % mode)

    logger.info('DONE')


if __name__ == "__main__":
    assert len(sys.argv) == 2, 'Specify 1 positional argument -- config file'
    config_fn = sys.argv[1]
    config_d = load_config(config_fn)
    run(config_d)
