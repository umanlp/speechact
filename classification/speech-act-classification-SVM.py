import os
import sys
import pandas as pd
import configparser
import logging
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss


def preprocess_text(tokens_list):
    punc_list = string.punctuation + "„" + "“"
    no_punc = [el for el in tokens_list if el not in punc_list]
    no_punc_lower = [el.lower() for el in no_punc]
    no_punc_lower_stems = [stemmer.stem(el) for el in no_punc_lower]

    return no_punc_lower_stems


def evaluate_svm(multilabel_classifier, X_test, y_test):
    y_pred = multilabel_classifier.predict(X_test)

    f1_score_per_class = dict(zip(unique_labels, f1_score(y_true=y_test, y_pred=y_pred, average=None) * 100))
    f1_micro_average = f1_score(y_true=y_test, y_pred=y_pred, average='micro') * 100
    roc_auc = roc_auc_score(y_test, y_pred, average='micro') * 100
    accuracy = accuracy_score(y_test, y_pred) * 100
    hamming_loss_value = hamming_loss(y_test, y_pred) * 100

    logging.info("SVM results:\n"
                 "F1 per class:\n{}\n"
                 "Micro-F1: {} / ROC-AUC: {} / Subset accuracy: {} / Hamming loss: {}".format(f1_score_per_class,
                                                                                              round(f1_micro_average,
                                                                                                    2),
                                                                                              round(roc_auc, 2),
                                                                                              round(accuracy, 2),
                                                                                              round(hamming_loss_value,
                                                                                                    2)))



if __name__ == '__main__':
    config_parser = configparser.ConfigParser()
    config_parser.read(sys.argv[1])

    if not os.path.exists(config_parser["BASE"]["output_dir"]):
        os.makedirs(config_parser["BASE"]["output_dir"])

    logging.basicConfig(filename=config_parser["BASE"]["output_dir"] + config_parser["BASE"]["log_filename"],
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

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
    segment_df_document = pd.read_pickle(input_dataframe_filename)
    logging.info("Loaded input dataframe from {}.".format(input_dataframe_filename))

    stemmer = SnowballStemmer("german")
    segment_df_document["utterance_token_list_clean"] = segment_df_document["utterance_token_list"].map(lambda x: preprocess_text(x))
    segment_df_document["utterance_token_string_clean"] = segment_df_document["utterance_token_list_clean"].map(lambda x: " ".join(x))

    vectorizer = TfidfVectorizer(
        analyzer="word", max_df=0.3, min_df=10, ngram_range=(1, 2), norm="l2"
    )
    vectorizer.fit(segment_df_document["utterance_token_string_clean"])

    train = segment_df_document[segment_df_document["split"] == "train"]
    dev = segment_df_document[segment_df_document["split"] == "dev"]
    test = segment_df_document[segment_df_document["split"] == "test"]


    X_train, X_test = vectorizer.transform(train["utterance_token_string_clean"]), vectorizer.transform(
        dev["utterance_token_string_clean"])
    y_train, y_test = train[unique_labels].to_numpy(), dev[unique_labels].to_numpy()
    svm = LinearSVC(dual=False)
    multilabel_classifier = MultiOutputClassifier(svm).fit(X_train, y_train)

    logging.info("Evaluating SVM on dev set.")
    evaluate_svm(multilabel_classifier, X_test, y_test)

    do_test: bool = config_parser["BASE"].getboolean("do_test")
    if do_test:
        X_test = vectorizer.transform(test["utterance_token_string_clean"])
        y_test = test[unique_labels].to_numpy()
        logging.info("Evaluating SVM on test set.")
        evaluate_svm(multilabel_classifier, X_test, y_test)
