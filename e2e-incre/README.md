# E2E NLG Challenge submission 

Please use the following citation:

```
@misc{puzikov2018_e2e-nlg-challenge,
 author = {Puzikov, Yevgeniy and Gurevych, Iryna},
 title = {{E2E} {NLG} Challenge: Neural Networks Vs. Templates},
 howpublished = {{E2E} {NLG} Challenge System Descriptions},
 year = 2018,
 note = {\url{http://www.macs.hw.ac.uk/InteractionLab/E2E/#papers}},
}
```

> **Abstract:**  
  E2E NLG Challenge is a shared task on generating restaurant
  descriptions from sets of key-value pairs.
  This paper describes the results of our participation in the challenge. 
  We develop a simple, yet effective neural encoder-decoder model
  which produces fluent restaurant descriptions
  and outperforms a strong baseline.
  We further analyze the data provided by the organizers and conclude
  that the task can also be approached with a template-based model
  developed in just a few hours.

Contact person: Yevgeniy Puzikov, puzikov@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, 
if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose 
of giving additional background details on the respective publication. 

## Background info

* Official website: http://www.macs.hw.ac.uk/InteractionLab/E2E/
* Submission deadline: 31 October 2017
* Evaluation protocol: 
    - automatic metrics for system development
    - final (human) evaluation by crowd workers and experts
     
## Project structure

The repository contains code for an MLP-based encoder-decoder model and a template-based deterministic system:

* `run_experiment.py`: main script to run
* `config_e2e_MLP_train.yaml` and `config_e2e_MLP_predict.yaml`: configuration files to use with the script above
* `components/`: NN components and the template model
* `predictions/`:
    * `e2e_model_MLP_seedXXX`: 20 folders with predictions and scores from the NN model (one per different random seed)
    * `model-t_predictions.txt` -- predictions of the template-based model
    * `aggregate.py` -- a script to aggregate NN model scores

## Requirements

* 64-bit Linux versions
* Python 3 and dependencies:
    * PyTorch v0.2.0
    * Progressbar2 v3.18.1
* Python 2

## Installation

* Install Python3 dependencies:

```
$ conda install pytorch torchvision cuda80 -c soumith 
$ conda install progressbar2
```

* Python2 dependencies are needed only to run the official evaluation scripts.
See installation instructions [here][3].

## Running the experiments

### Preparation

* Step 1

The repository contains two template yaml files 
for training Model-D and using it later for prediction.

**Before** using the files, run:

```
$ envsubst < config.yaml > my-config.yaml
```

This will replace shell format strings (e.g, $HOME) in your .yaml
files with the corresponding environment variables' values 
(see [this page][1] for details). Use the *my-config.yaml* for the experiments.

* Step2 

Modify `PYTHON2` and `E2E_METRCIS_FOLDER` variables in the following file:

`components/evaluator/eval_scripts/run_eval.sh`

This shell script is calling the [external evaluation tools][3].
`PYTHON2` denotes a specific python environment with all the necessary dependencies installed.
`E2E_METRICS_FOLDER` denotes the cloned repository with the aforementioned tools.

### Training models

* **Model-D**:
    1. Adjust data paths and hyper-parameter values in the config file (*my_config.yaml*, as a running example).
    
    2. Run the following command:  
        
    ```
    $ python run_experiment.py my_config.yaml
    ```
    
    3. After the experiment, a folder will be created in the directory specified 
    by the *experiments_dir* field of *my_config.yaml* file.
    This folder should contain the following files:
        - experiment log (*log.txt*)
        - model weights and development set predictions for each training epoch 
        (*weights.epochX*, *predictions.epochX*)    
        - a csv file with scores and train/dev losses for each epoch (*scores.csv*)
        - configuration dictionary in json format (*config.json*)
	    - pdf files with learning curves (optional)
    
    4. If you use a model for prediction 
    (by setting "predict" as the value for the *mode* field in the config file and 
    specifying model path in *model_fn*), the predictions done by the loaded model will be
    stored in:
        - $model_fn.devset.predictions.txt
        - $model_fn.testset.predictions.txt
   
* **Model-T**:
    
    To make predictions on *filename.txt*, run the following command:
    
    ```
    $ python components/template-baseline.py filename.txt MODE
    ```
    Here, *filename.txt* is either devset or testset CSV file; 
    *MODE* can be either 'dev' or 'test'. 
    
    Model-T's predictions are saved in *filename.txt.predicted*.

## Evaluation

*./predictions* contains prediction files for 20 instances of Model-D, 
trained with different values of the random seed. 
Note that those are predictions scored highest epoch-wise for each model.
The folder also contains the predictions of Model-T (also on dev set) and
a Python script to aggregate the results.

Navigate to *./predictions/* and run:

```
python aggregate.py */scores.csv
```

This will output mean scores averaged over 20 runs 
(with standard deviation and some other useful statistics).

### Expected results

After running the experiments, you should expect the following results (development set):

Metric  	| TGen 	|    Model-D 	        | Model-T
:---------	|---------:	|-------:	        |---------:
BLEU    	|   0.6925 	| **0.7128** (+-0.013)	|   0.6051
NIST    	|   8.4781 	| **8.5020** (+-0.092) 	|   7.5257
METEOR  	|   0.4703 	| **0.4770** (+-0.012)	|   0.4678
ROUGE-L 	|   0.7257 	| **0.7378** (+-0.015)	|   0.6890
CIDEr   	|   2.3987 	| **2.4432** (+-0.088) 	|   1.6997

- TGen - baseline from the organizers
- Model-D - data-driven model (enc-dec model with an MLP as encoder)
- Model-T - template-based system


[1]: https://www.gnu.org/software/gettext/manual/gettext.html#sh_002dformat
[2]: http://www.macs.hw.ac.uk/InteractionLab/E2E/data/baseline-output.txt
[3]: https://github.com/tuetschek/e2e-metrics
