# -*- coding: utf-8 -*-

"""
BERTForSequenceClassification baseline for multi-label speech act classification.
This script:
1. either:
    - loads a .pkl file provided in the variable input_dataframe in the .conf file
    - or loads the annotated BT data and creates a dataset "from scratch" using annotations_directory and
    splits_directory provided in the .conf file
2. fine-tunes OR evaluates a multi-label BERTForSequenceClassification model

We use the column 'utterance_token_list' (or 'synfeat' instead, if available) as input to the BERT classifier.
"""
import logging
import os
import sys

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import configparser
import random
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from transformers import EvalPrediction
from transformers import set_seed

from speech_acts_utils import tokenize_and_align_token_type_ids
from speech_acts_utils import tokenize_no_alignment
from speech_acts_utils import preprocess_labels
from speech_acts_utils import print_dataset_info
from speech_acts_utils import add_sentence_tokens

def logits_to_probas(predictions, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probas = sigmoid(torch.Tensor(predictions)).numpy()  # logits turned into probs for each of the 15 classes
    y_pred = np.zeros(probas.shape)  # fill with zeros first
    y_pred[np.where(probas >= threshold)] = 1  # replace 0 with 1 where prob above threshold

    return y_pred, probas

def multi_label_metrics(predictions, labels):
    # first, turn raw logits into probabilities for each class using sigmoid function:
    y_pred, probas = logits_to_probas(predictions, threshold=0.5)
    # then, compute metrics:
    y_true = labels
    precision_per_class = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    recall_per_class = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    hamming_loss_value = hamming_loss(y_true, y_pred)
    # return as dictionary
    metrics = {'precision_per_class': precision_per_class,
               'recall_per_class': recall_per_class,
               'f1_per_class': f1_per_class,
               'f1_micro': f1_micro_average,
               'accuracy': accuracy,
               'hamming_loss': hamming_loss_value}
    return metrics

def compute_metrics(p: EvalPrediction):
    # EvalPrediction is a transformers helper class for the Trainer
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

# HYPERPARAMETER SEARCH
def hs_objective(metrics):
    return metrics["eval_loss"]
    # by default, evaluate() adds the prefix "eval_" to metrics returned by compute_metrics()
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 4, 5, 6])
    }
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,
                                                              problem_type="multi_label_classification",
                                                              num_labels=len(unique_labels),
                                                              id2label=id2label,
                                                              label2id=label2id)

def majority_baseline(eval_dataset, majority_class, label2id):
    y_true = eval_dataset["labels"]
    majority_label_position = label2id[majority_class]
    y_pred = torch.zeros_like(y_true)
    mask = torch.tensor([majority_label_position])
    y_pred[:, mask] = 1.0

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    hamming_loss_value = hamming_loss(y_true, y_pred)

    return {'f1_micro': f1_micro_average, 'accuracy': accuracy, 'hamming_loss': hamming_loss_value}

def train_model(do_hyperparam_search=False):
    print("Loading model from HuggingFace checkpoint.")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(unique_labels),
                                                               id2label=id2label,
                                                               label2id=label2id)

    if do_hyperparam_search:
        logging.info("Start hyperparameter search.")
        trainer = Trainer(
            model_init=model_init,  # hyperparam search requires model_init
            args=args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics
        )
        best_run = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=20,
            compute_objective=hs_objective,  # defaults to validation loss
            study_name="BERT_speech_act_hyperparam_search"
        )
        string_best_hyperparam = "Hyperparameter search done. Best hyperparameters:"
        print(string_best_hyperparam)
        for n, v in best_run.hyperparameters.items():
            print(n, v)
            string_best_hyperparam += "\n{}: {}".format(n, v)
            setattr(trainer.args, n, v)
        logging.info("Best hyperparameters:")
        logging.info(string_best_hyperparam)
        logging.info("Start training with best hyperparameters. Seed: {}".format(SEED))
        print("Start training with best hyperparameters.")

    else:
        if consistency_check_mode:
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset["all"],
                eval_dataset=tokenized_dataset["all"],
                compute_metrics=compute_metrics
            )
        else:
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                compute_metrics=compute_metrics
            )

        logging.info("Start training. Seed: {}".format(SEED))

    # TRAIN MODEL
    trainer.train()
    logging.info("Training done.")
    do_save_model: bool = config_parser["BASE"].getboolean("do_save_model")
    if do_save_model:
        trainer.save_model()
        logging.info("Model saved at: {}".format(config_parser["BASE"]["output_dir"]))
    training_log = pd.DataFrame(trainer.state.log_history)
    logging.info("Training results:\n{}".format(training_log.to_string()))

    logging.info("Evaluation on development set:")
    dev_results = trainer.evaluate()

    logging.info(
        "Development set results:\nPrecision per class: {}\nRecall per class: {} / F1 per class: {} / Micro-F1: {} / Subset accuracy: {} / Hamming loss: {}".format(
            dev_results["eval_precision_per_class"],
            dev_results["eval_recall_per_class"],
            dev_results["eval_f1_per_class"],
            dev_results["eval_f1_micro"],
            dev_results["eval_accuracy"],
            dev_results["eval_hamming_loss"]))

    return trainer, dev_results

def evaluate_model(segment_df_full, do_test=False, consistency_check_mode=False):

    if consistency_check_mode:
        eval_dataset = "all"
        print("CAUTION: EVALUATING ON ENTIRE DATASET")
        logging.info("CAUTION: EVALUATING ON ENTIRE DATASET")
    else:
        if do_test:
            logging.info("Evaluation on test set.")
            eval_dataset = "test"
            split = "test"
        else:
            logging.info("Evaluation on validation set.")
            eval_dataset = "validation"
            split = "dev"
    predictions, gold_labels, results = trainer.predict(tokenized_dataset[eval_dataset])
    f1_per_class = dict(zip(unique_labels, results["test_f1_per_class"]))
    # the order in unique_labels is fixed (because it's a list) and corresponds to label2id
    logging.info(
        "Results:\nF1 per class: {}\nMicro-F1: {} / Subset accuracy: {} / Hamming loss: {}".format(
            f1_per_class,
            results["test_f1_micro"],
            results["test_accuracy"],
            results["test_hamming_loss"]))

    y_pred_dev, probas = logits_to_probas(predictions, threshold=0.5)
    if consistency_check_mode:
        segment_df_full_eval = segment_df_full.copy().reset_index(drop=True) # things silently get messed up if you don't reset the index
    else:
        segment_df_full_eval = segment_df_full[segment_df_full["split"] == split].reset_index(drop=True)
    pred_df = pd.DataFrame(y_pred_dev, columns=unique_labels)
    confidences_df = pd.DataFrame(probas, columns=unique_labels)
    pred_df["predicted_labels"] = pd.DataFrame(pred_df.apply(lambda x: x[x == 1.0].index.values.tolist(), axis=1),
                                    columns=["predicted_labels"])
    segment_df_full_eval["predicted_labels"] = pd.DataFrame(pred_df.apply(lambda x: x[x == 1.0].index.values.tolist(), axis=1),
                                    columns=["predicted_labels"])
    segment_df_full_eval["confidences"] = pd.DataFrame(confidences_df.apply(lambda x: x[x >= 0.5].values.tolist(), axis=1), columns=["confidences"])

    predictions_df = segment_df_full_eval[["utterance_token_list", "document", "utterance_labels", "predicted_labels",
                                           "confidences"]]

    return results, predictions_df

if __name__ == '__main__':

    # LOGGING THINGS
    config_parser = configparser.ConfigParser()
    config_parser.read(sys.argv[1])

    # SET SEEDS
    SEED = int(config_parser['PARAM']['seed'])
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)  # transformers

    if not os.path.exists(config_parser["BASE"]["output_dir"]):
        os.makedirs(config_parser["BASE"]["output_dir"])

    logging.basicConfig(filename=config_parser["BASE"]["output_dir"] + config_parser["BASE"]["log_filename"],
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # GPU THINGS
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config_parser["BASE"]["gpu_node"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: {}".format(device))

    merge_questions: bool = config_parser["BASE"].getboolean("do_merge_questions")
    if merge_questions:
        # important: this has to be a list (and not a set) for order purposes
        unique_labels = ["Accusation", "Evaluation", "Request", "Promise", "Bad-outcome",
                         "Report", "Self-representation", "Support",
                         "Demand", "Rejection", "Question-All", "Expressive", "I-S-Humour", "Macro"]
        logging.info("Merging Rhetorical-question and Question instances into a single Question-All class.")
    else:
        # important: this has to be a list (and not a set) for order purposes
        unique_labels = ["Accusation", "Evaluation", "Request", "Promise", "Bad-outcome",
                         "Report", "Rhetorical\\_question", "Self-representation", "Support",
                         "Demand", "Rejection", "Question", "Expressive", "I-S-Humour", "Macro"]
        logging.info("Keeping Rhetorical-question and Question instances separate.")

    input_dataframe_filename = config_parser["BASE"]["input_dataframe"]
    file_ending = input_dataframe_filename.split(".")[-1]
    if file_ending == "pkl":
        segment_df_document = pd.read_pickle(input_dataframe_filename)
        logging.info("Loaded input dataframe from {}.".format(input_dataframe_filename))
        if "synfeat" in segment_df_document.columns:
            segment_df_document.drop(columns=["utterance_token_list"], inplace=True)
            segment_df_document.rename(columns={"synfeat": "utterance_token_list"}, inplace=True)
            synfeat=True
            logging.info("Found 'synfeat' column in input dataframe. Using this column as input to BERT.")
        else:
            synfeat=False
            logging.info("Using 'utterance_token_list' column as input to BERT.")
    else:
        sys.exit("Input dataframe must be .pkl. Please edit the variable input_dataframe in the conf file.")

    # ADD SENTENCE CONTEXT
    context_mode: bool = config_parser["PARAM"].getboolean("context_mode")
    logging.info("Context mode is set to {}.".format(context_mode))
    if context_mode:
        logging.info("Adding context tokens to dataset, this takes several minutes.")
        segment_df_document = add_sentence_tokens(segment_df_document, synfeat=synfeat)

    if context_mode:
        columns_to_keep = unique_labels + ["sentence_token_type_ids", "utterance_labels", "sentence_token_list", "split"]
    else:
        columns_to_keep = unique_labels + ["utterance_labels", "utterance_token_list", "split"]
    segment_df = segment_df_document[columns_to_keep]
    consistency_check_mode: bool = config_parser["BASE"].getboolean("do_consistency_check_mode")
    majority_class = print_dataset_info(segment_df, consistency_check_mode)

    # ENCODE THE DATASET: TOKENIZATION AND MULTI-LABEL SETUP
    CHECKPOINT = config_parser["BASE"]["pretrained_bert_model"]
    train_dataset = Dataset.from_pandas(segment_df[segment_df["split"] == "train"])
    dev_dataset = Dataset.from_pandas(segment_df[segment_df["split"] == "dev"])
    test_dataset = Dataset.from_pandas(segment_df[segment_df["split"] == "test"])

    if consistency_check_mode:
        logging.info("CAUTION: Consistency check mode on: model is trained and evaluated on entire dataset (train, "
                     "dev and test!")
        segment_df["split"] = "all"
        bt_dataset = DatasetDict({"all": Dataset.from_pandas(segment_df)})
    else:
        bt_dataset = DatasetDict({"train": train_dataset, "validation": dev_dataset, "test": test_dataset})

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    if context_mode:
        tokenized_aligned_dataset = bt_dataset.map(tokenize_and_align_token_type_ids,
                                                   batched=True, fn_kwargs={"tokenizer": tokenizer},
                                                   remove_columns=["sentence_token_type_ids",
                                                                   "sentence_token_list", "split"])
    else:
        try:
            tokenized_aligned_dataset = bt_dataset.map(tokenize_no_alignment,
                                                       batched=True, fn_kwargs={"tokenizer": tokenizer},
                                                       remove_columns=["old_row_indices",
                                                                       "utterance_token_list", "split"])
        except ValueError:
            tokenized_aligned_dataset = bt_dataset.map(tokenize_no_alignment,
                                                       batched=True, fn_kwargs={"tokenizer": tokenizer},
                                                       remove_columns=["__index_level_0__",
                                                                       "utterance_token_list", "split"])

    columns_to_keep = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    if consistency_check_mode:
        columns_to_remove = [col for col in tokenized_aligned_dataset["all"].column_names if
                             col not in columns_to_keep]
    else:
        columns_to_remove = [col for col in tokenized_aligned_dataset["train"].column_names if
                             col not in columns_to_keep]

    tokenized_dataset = tokenized_aligned_dataset.map(preprocess_labels, batched=True,
                                                      fn_kwargs={"unique_labels": unique_labels},
                                                      remove_columns=columns_to_remove)

    tokenized_dataset.set_format("torch")

    # for debugging purposes
    if not consistency_check_mode:
        example = tokenized_dataset["train"][0]

    # OPTIONAL TOY SETTING FOR DEBUGGING
    toy_setting: bool = config_parser["BASE"].getboolean("do_toy")
    if toy_setting:
        toy_size = 50
        logging.info("TOY SETTING! Training set reduced to first {} instances.".format(toy_size))
        tokenized_dataset["train"] = tokenized_dataset["train"].select(range(toy_size))

    # DEFINE TRAINING ARGUMENTS AND METRICS
    args = TrainingArguments(
        output_dir=config_parser["BASE"]["output_dir"],
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=float(config_parser['PARAM']['learning_rate']),
        per_device_train_batch_size=int(config_parser['PARAM']['batch_size']),
        per_device_eval_batch_size=8,
        num_train_epochs=int(config_parser['PARAM']['epochs']),
        optim=config_parser['PARAM']['optimizer'],
        weight_decay=float(config_parser['PARAM']['weight_decay']),
        warmup_ratio=float(config_parser['PARAM']['warmup_ratio']),
        load_best_model_at_end=False,
        log_level="info",
        logging_strategy="epoch",
        seed=SEED,
        # group_by_length=True,
    )

    # TRAIN AND/OR EVALUATE ON DEV/TEST SET
    do_train: bool = config_parser["BASE"].getboolean("do_train")
    do_hyperparam_search: bool = config_parser["BASE"].getboolean("do_hyperparameter_search")
    do_test: bool = config_parser["BASE"].getboolean("do_test")
    if do_train:
        id2label = {idx: label for idx, label in enumerate(unique_labels)}
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        trainer, dev_results = train_model(do_hyperparam_search)
        if consistency_check_mode:
            _, predictions_df = evaluate_model(segment_df_document, do_test=False,
                                               consistency_check_mode=True) # evaluate on the ENTIRE dataset
            predictions_df_filename = config_parser["BASE"]["output_dir"] + "BERT_no_context_consistency_check.tsv"
        else:
            _, predictions_df = evaluate_model(segment_df_document, do_test=False,
                                               consistency_check_mode=False)  # evaluate on dev set per default
            predictions_df_filename = config_parser["BASE"]["output_dir"] + "preds_dev_set.tsv"
        with open(predictions_df_filename, "w") as outf:
            predictions_df.to_csv(outf, sep="\t", index=False)

    else: # don't train, load fine-tuned model instead
        do_average_runs: bool = config_parser["BASE"].getboolean("do_average_runs")
        if do_average_runs: # average multiple runs
            runs_paths_list = config_parser["BASE"]["runs_paths"].split(",")
            dev_results_all_runs = []
            print("Evaluating {} fine-tuned models".format(len(runs_paths_list)))
            logging.info("Evaluating {} fine-tuned models".format(len(runs_paths_list)))
            for fine_tuned_model_path in runs_paths_list:
                print("Loading fine-tuned model from", fine_tuned_model_path)
                logging.info("Loading fine-tuned model from {}".format(fine_tuned_model_path))
                model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
                trainer = Trainer(
                    model,
                    args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["validation"],
                    compute_metrics=compute_metrics
                )
                dev_results, predictions_df = evaluate_model(segment_df_document, do_test, consistency_check_mode)
                out_path = fine_tuned_model_path[:-1].split("/")[-1]
                if do_test:
                    predictions_df_filename = config_parser["BASE"]["output_dir"] + out_path + "_preds_test_set.tsv"
                else:
                    predictions_df_filename = config_parser["BASE"]["output_dir"] + out_path + "_preds_dev_set.tsv"
                with open(predictions_df_filename, "w") as outf:
                    predictions_df.to_csv(outf, sep="\t", index=False)
                dev_results_all_runs.append(dev_results)
            all_precision_scores_per_class = [result["test_precision_per_class"] for result in dev_results_all_runs]
            all_recall_scores_per_class = [result["test_recall_per_class"] for result in dev_results_all_runs]
            all_f1_scores_per_class = [result["test_f1_per_class"] for result in dev_results_all_runs]

            df_f1_scores_per_class = pd.DataFrame(all_f1_scores_per_class, columns=unique_labels)
            df_f1_scores_per_class.loc["mean"] = df_f1_scores_per_class.mean()
            df_f1_scores_per_class.loc["std"] = df_f1_scores_per_class.drop(labels=["mean"]).std(ddof=0)
            # we have the entire population so no need for Bessel's correction

            df_precision_scores_per_class = pd.DataFrame(all_precision_scores_per_class, columns=unique_labels)
            df_precision_scores_per_class.loc["mean"] = df_precision_scores_per_class.mean()
            df_precision_scores_per_class.loc["std"] = df_precision_scores_per_class.drop(labels=["mean"]).std(ddof=0)

            df_recall_scores_per_class = pd.DataFrame(all_recall_scores_per_class, columns=unique_labels)
            df_recall_scores_per_class.loc["mean"] = df_recall_scores_per_class.mean()
            df_recall_scores_per_class.loc["std"] = df_recall_scores_per_class.drop(labels=["mean"]).std(ddof=0)

            df_f1_scores_per_class = df_f1_scores_per_class * 100  # for readability
            df_precision_scores_per_class = df_precision_scores_per_class * 100
            df_recall_scores_per_class = df_recall_scores_per_class * 100
            print("F1 scores per class for all runs:")
            print(df_f1_scores_per_class.to_markdown())
            logging.info("F1 scores per class for all runs:\n{}".format(df_f1_scores_per_class.to_markdown()))
            results_string = "{:.2f} ({:.2f})"

            df_scores_per_class = df_f1_scores_per_class.T[["mean", "std"]]
            df_scores_per_class.rename(columns={"mean": "mean_f1", "std": "std_f1"}, inplace=True)
            df_scores_per_class['f1'] = [results_string.format(x, y) for x, y in
                                                zip(df_scores_per_class['mean_f1'],
                                                    df_scores_per_class['std_f1'])]
            df_scores_per_class["mean_precision"] = df_precision_scores_per_class.T["mean"]
            df_scores_per_class["std_precision"] = df_precision_scores_per_class.T["std"]
            df_scores_per_class['precision'] = [results_string.format(x, y) for x, y in
                                                zip(df_scores_per_class['mean_precision'],
                                                    df_scores_per_class['std_precision'])]
            df_scores_per_class["mean_recall"] = df_recall_scores_per_class.T["mean"]
            df_scores_per_class["std_recall"] = df_recall_scores_per_class.T["std"]
            df_scores_per_class['recall'] = [results_string.format(x, y) for x, y in
                                                zip(df_scores_per_class['mean_recall'],
                                                    df_scores_per_class['std_recall'])]
            logging.info("LaTex table:\n{}".format(df_scores_per_class.to_latex(columns=["precision", "recall", "f1"])))

            all_f1_micro = [result["test_f1_micro"]*100 for result in dev_results_all_runs]
            all_accuracy = [result["test_accuracy"]*100 for result in dev_results_all_runs]
            all_hamming_loss = [result["test_hamming_loss"]*100 for result in dev_results_all_runs]
            logging.info("Micro-F1:\t\t\taveraged: {}% / std dev: {}".format(round(np.mean(all_f1_micro), ndigits=2),
                                                                             round(np.std(all_f1_micro), ndigits=3)))
            logging.info("Subset accuracy:\taveraged: {}% / std dev: {}".format(round(np.mean(all_accuracy), ndigits=2),
                                                                                round(np.std(all_accuracy), ndigits=3)))
            logging.info("Hamming loss:\t\taveraged: {}% / std dev: {}".format(round(np.mean(all_hamming_loss), ndigits=2),
                                                                               round(np.std(all_hamming_loss), ndigits=3)))
        else: # evaluate a single model
            fine_tuned_model_path = config_parser["BASE"]["fine_tuned_model_path"]
            print("Loading fine-tuned model from", fine_tuned_model_path)
            logging.info("Loading fine-tuned model from {}".format(fine_tuned_model_path))
            model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
            if consistency_check_mode:
                trainer = Trainer(
                    model,
                    args,
                    train_dataset=tokenized_dataset["all"],
                    eval_dataset=tokenized_dataset["all"],
                    compute_metrics=compute_metrics
                )
                predictions_df_filename = config_parser["BASE"]["output_dir"] + "BERT_no_context_consistency_check.tsv"
            else:
                trainer = Trainer(
                    model,
                    args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["validation"],
                    compute_metrics=compute_metrics
                )
                out_path = fine_tuned_model_path[:-1].split("/")[-1]
                if do_test:
                    predictions_df_filename = config_parser["BASE"]["output_dir"] + out_path + "_preds_test_set.tsv"
                else:
                    predictions_df_filename = config_parser["BASE"]["output_dir"] + out_path + "_preds_dev_set.tsv"
            results, predictions_df = evaluate_model(segment_df_document, do_test, consistency_check_mode)

            with open(predictions_df_filename, "w") as outf:
                predictions_df.to_csv(outf, sep="\t", index=False)

    # OPTIONAL MAJORITY BASELINE
    do_majority_baseline: bool = config_parser["BASE"].getboolean("do_majority_baseline")
    if do_majority_baseline:
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        if do_test:
            eval_dataset = tokenized_dataset["test"]
            logging.info("Evaluating majority baseline on test set.")
        else:
            eval_dataset = tokenized_dataset["validation"]
            logging.info("Evaluating majority baseline on dev set.")
        majority_results = majority_baseline(eval_dataset, majority_class, label2id)
        print("Scores for majority baseline:\nMicro-F1: {} / Subset accuracy: {} / Hamming loss: {}".format(majority_results["f1_micro"],
                                                                                                            majority_results["accuracy"],
                                                                                                            majority_results["hamming_loss"]))

        logging.info("Scores for majority baseline:\nMicro-F1: {} / Subset accuracy: {} / Hamming loss: {}".format(majority_results["f1_micro"],
                                                                                                                   majority_results["accuracy"],
                                                                                                                   majority_results["hamming_loss"]))

