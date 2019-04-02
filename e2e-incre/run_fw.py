import importlib
import os, json
import sys

from components.utils.config import load_config, fix_seed
from components.utils.log import set_logger
from components.utils.serialization import load_model, make_model_dir
from components.utils.serialization import save_config, save_predictions_txt

def save_beam_fw(fw_probs, decode_snts, beam_size, filename):
    with open(filename, "w") as outstream:
        pp, pp_u = [], set()
        for dec_idx in range(len(decode_snts[0])):
            dec_cur, fw_cur = [], []
            for beam_idx in range(beam_size):
                tmp_cur = ' '.join(decode_snts[beam_idx][dec_idx])
                #assert tmp_cur not in dec_cur
                pp.append(tmp_cur), pp_u.add(tmp_cur)
                dec_cur.append( tmp_cur )
                fw_cur.append( fw_probs[beam_idx][dec_idx]  )
            outstream.write("%s\t%s\n" % (json.dumps(dec_cur), json.dumps(fw_cur)))
        print("unique ratio:", len(list(pp_u)) * 100.0 / len(pp))

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
    #print(len(data.lexicalizations['test']))
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
        #print(model.state_dict())
        model = model.to('cuda')
        id2word = data.vocab.id2tok
        beam_size = None #10
        alpha = 0.2 # hyperparameter to control the ratio of incremental paragmatics
        #"""
        if 'dev' in data.fnames:
            logger.info("Predicting on dev data")
            # print(len(data.uni_mr['dev']), len(data.dev[0]), len(data.lexicalizations['dev']))
            dec_snt_beam, fw_beam = [], []
            predicted_ids, fw_beam = evaluator.evaluate_model(model, data.dev[0], data.uni_mr['dev'], beam_size=beam_size, alpha=alpha)
            data_lexicalizations = data.lexicalizations['dev']
            # print(len(predicted_ids), len(data_lexicalizations))
            predicted_snts = evaluator.lexicalize_predictions(predicted_ids, data_lexicalizations, id2word)
            if beam_size is None:
                save_predictions_txt(predicted_snts, '%s.devset.predictions.txt_incre_%.1f_new' % (model_fname, alpha))
            else:
                save_beam_fw(fw_beam, dec_snt_beam, beam_size, '%s.devset.recs.txt' % model_fname)
            exit()
        #"""    
        if 'test' in data.fnames:
            logger.info("Predicting on test data")
            print(len(data.test[0]))
            predicted_ids, attention_weights = evaluator.evaluate_model(model, data.test[0], data.uni_mr['test'], beam_size=beam_size, alpha=alpha)
            data_lexicalizations = data.lexicalizations['test']
            print(len(predicted_ids), len(data_lexicalizations))
            predicted_snts = evaluator.lexicalize_predictions(predicted_ids,
                                                              data_lexicalizations,
                                                              id2word)
            save_predictions_txt(predicted_snts, '%s.testset.predictions.txt_inre_%.1f' % (model_fname, alpha))

    else:
        logger.warning("Check the 'mode' field in the config file: %s" % mode)

    logger.info('DONE')


if __name__ == "__main__":
    assert len(sys.argv) == 2, 'Specify 1 positional argument -- config file'
    config_fn = sys.argv[1]
    config_d = load_config(config_fn)
    run(config_d)
