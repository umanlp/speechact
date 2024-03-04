# Speech act classification

## Repository description
This repository contains the code for our paper "How to do politics with words: Investigating speech acts in parliamentary debates".
The code can be used to replicate results from the paper and access model predictions.

If you use this code, please cite our paper:
```
TODO: bibtex
```

## Annotation guidelines

The file `Guidelines_speechacts.en.pdf` contains the annotation guidelines that we used to annotate speech acts in parliamentary debates.

## Speech act classification

### Setup

This project was tested with python 3.11.

The requirements for speech act classification can be found in the file `classification/requirements.txt`.

### Usage

In general, the following script can be used to train or evaluate a speech act classification model:
```
python3 speech-act-classification-BERT.py [your-configuration-file.conf]
```

The behaviour of the script depends on the contents of the provided .conf file. Different .conf files for evaluation and training are presented below.

The script always generates a logfile with information about the settings that you defined in the .conf file.

We also provide the code that we used for our SVM-based baseline:
```
python3 speech-act-classification-SVM.py config/svm-speech-act-classification.conf
```

### Model and prediction files

We make one run of our best speech act classification model, BERT-large-context-run-x (Micro-F1 reported in table 5 of the paper: 84.01%/81.96% dev/test), publicly available. 
Due to space constraints in this repository, the model has to be downloaded from the following source: `TODO: link/to/model`

Please place the model's directory directly in this `classification` folder or change the path accordingly in the .conf file.

We make BERT-based model predictions on the dev and test set available in the folder `predictions`:
- BERT-base w/o context (79.99% Micro-F1 on the dev set): `BERT-base-all-runs/`
- BERT-base w/ context (80.18% Micro-F1 on the dev set): `BERT-base-context-all-runs/`
- BERT-large w/ context (84.01% Micro-F1 on the dev set): `BERT-large-context-all-runs/`

The dataset used in classification experiments is available as `dataset.pkl` or in TSV-format (`unzip dataset.zip`).

### Train a BERT-based model

First, modify `train-BERT_context.conf` to your liking. Change the following variables:
- `pretrained_bert_model`: the name of the underlying pre-trained BERT model in the `transformers` library
- `output_dir`: path where the model will be saved
- `context_mode`: boolean indicating whether to use the `BERT_context` setting, as described in the paper.
- the hyperparameters in the section `[PARAM]`, if needed

Then, run the script:
```
python3 speech-act-classification-BERT.py train-BERT_context.conf
```

This trains a single run and saves it in the path provided in `output_dir`.

### Evaluate a BERT-based model / Generate predictions

To evaluate one specific run of a model, proceed as follows:
- Change the model path in the variable `fine_tuned_model_path` in the .conf file.
- Run the script: `python3 speech-act-classification-BERT.py config/eval_BERT_single_run.conf`

To evaluate several runs of one model, proceed as follows:
- Change the model path in the variable `runs_paths` in the .conf file.
- Run the script: `python3 speech-act-classification-BERT.py config/eval_BERT_multiple_runs_context.conf`

To run a model without context, set the variable `context_mode` in the .conf file to `False`.

By default, the script evaluates on the dev set. To evaluate on the test set instead, change the variable `do_test` to `True` in the .conf file.

The script writes predictions and the logfile to the output folder provided in the variable `output_dir`. You can change this variable to a desired output folder. If the folder does not exist, it will be created. If it does exist, its contents with identical filenames will be overwritten.


### More info about conf settings

The following table explains most of the variables in the .conf files:

| Variable | Explanation |
| -------- | ----------- |
| do_train | whether to train a model |
| do_save_model | whether to save the model that was trained |
| do_hyperparameter_search | whether to conduct hyperparameter search |
| do_test | whether to evaluate on the test set instead of the dev set (default) |
| do_toy | whether to limit the train set to few instances (for debugging) |
| do_majority_baseline | whether to also evaluate a majority baseline on the provided dataset |
| do_average_runs | whether to evaluate and average multiple runs provided in runs_paths below |
| do_consistency_check_mode | Caution: this trains and evaluates on the entire dataset |
| do_merge_questions | whether rhetorical_question and question are merged to a question_all class |
| do_add_punc | whether end of sentence punctuation is to be added (ignore this if you use input_dataframe below) |
| splits_directory | path to directory containing information about splits (ignore this if you use input_dataframe below) |
| annotations_directory | path to directory containing the inception files (ignore this if you use input_dataframe below) |
| input_dataframe | path to the pickle dataframe containing the entire dataset |
| gpu_node | which GPU node to use on the cluster |
|  |
| pretrained_bert_model | the underlying pre-trained BERT model |
| fine_tuned_model_path | path to the folder containing the single fine-tuned model that you want to evaluate |
| runs_paths | paths to all the runs that you want to evaluate |
| output_dir | path to the directory where logfile and predictions will be written (will be created if does not exist, does overwrite) |
| log_filename | name of your logfile to be saved in output_dir |
| seed | seed for all things randomness (only relevant for training) |
| context_mode | whether to use the `BERT_context` setting: this gives the previous sentence, the sentence containing the utterance and the next sentence as input to BERT. The utterance to be classified is marked by setting the `token_type_ids` for corresponding to the value 1, and 0 for remaining context tokens. 

## Speech act segmentation

### Setup 

This project was tested with python 3.10.

The requirements for the speech act segmentation code can be found in the file `segmentation/README.md`.

### Usage

Follow the instructions in `segmentation/README.md` to create a virtual environment and install the required packages.
Then run the code with:

```
python src/segmenter.py config/segment.conf
```


## Analysis

We make the code to generate the figures in Section 6 of the paper available in the folder `analysis/`.

### Setup

This project was tested with python 3.11.

Requirements to run the code are provided in `analysis/analysis_requirements.txt`.

We provide two scripts:
- `speech_acts_stats.py`: This script generates plots for conflict, demand and request proportions for a given party (see Figures 3 and 4 in the paper).
- `speech_acts_all_parties_stats.py`: This scripts aggregates information about conflict and cooperative communications across all parties and generates the bar plot for Figure 2 in the paper.

### Usage

First, download the 6 input files for the scripts using the link provided in `analysis/party_pickles/link_to_files.txt` and place them in the folder `party_pickles/`. The files are too large to be stored in this GitHub repository.

Then, run one of both scripts:

```
python speech_acts_stats.py party_pickles/[party].pkl
```

and

```
python speech_acts_all_parties_stats.py party_pickles/
```
