import pandas as pd
import numpy as np

"""
Utils for speech-act-classification-BERT.py
"""

def print_dataset_info(df, consistency_check_mode):
    """
    Prints info about the provided dataset:
    - number of instances (gold segments) by train/dev/test split
    - class distributions
    - the majority class

    :param df: the dataset in pandas DataFrame format
    :return majority_class: the majority class in the training set
    """

    train = df[df["split"] == "train"].select_dtypes(include='bool')
    dev = df[df["split"] == "dev"].select_dtypes(include='bool')
    test = df[df["split"] == "test"].select_dtypes(include='bool')
    print("Number of instances (gold segments) per split:\n"
          "{} train, {} dev, {} test".format(len(train),
                                             len(dev),
                                             len(test)))

    print("Class distributions:")
    if consistency_check_mode:
        all_df = df.select_dtypes(include='bool')
        all_distr = all_df.apply(pd.Series.value_counts, normalize=True).loc[True]
    else:
        train_distr_norm = train.apply(pd.Series.value_counts, normalize=True).loc[True]
        train_distr = train.apply(pd.Series.value_counts, normalize=False).loc[True]
        dev_distr_norm = dev.apply(pd.Series.value_counts, normalize=True).loc[True]
        dev_distr = dev.apply(pd.Series.value_counts, normalize=False).loc[True]
        test_distr_norm = test.apply(pd.Series.value_counts, normalize=True).loc[True]
        test_distr = test.apply(pd.Series.value_counts, normalize=False).loc[True]
        all_distr = pd.concat([train_distr, train_distr_norm, dev_distr, dev_distr_norm,
                               test_distr, test_distr_norm],
                              keys=["train", "train_normalized", "dev",
                                    "dev_normalized", "test", "test_normalized"], axis=1)

    print(all_distr.to_latex())

    if consistency_check_mode:
        majority_class = all_distr.idxmax()
    else:
        majority_class = train_distr.idxmax()

    return majority_class

def add_sentence_tokens(segment_df_full, synfeat=False):
    sentence_token_list = []
    sentence_token_type_id_list = []
    for i, row in segment_df_full.iterrows():
        token_positions = [i for i in range(len(row["token_list"])) if row["custom_token_type_ids"][i] == 1]
        current_sent_start_id = row["token_sentence_ids"][token_positions[0]]
        previous_sent_id = current_sent_start_id - 1 if current_sent_start_id != 0 else None
        next_sent_id = current_sent_start_id + 1 if current_sent_start_id != max(row["token_sentence_ids"]) else None
        current_sent_positions = [i for i in range(len(row["token_list"])) if
                                  row["token_sentence_ids"][i] == current_sent_start_id]
        if previous_sent_id is not None:
            current_sent_positions = [i for i in range(len(row["token_list"])) if
                                      row["token_sentence_ids"][i] == previous_sent_id] + current_sent_positions
        if next_sent_id is not None:
            current_sent_positions = current_sent_positions + [i for i in range(len(row["token_list"])) if
                                                               row["token_sentence_ids"][i] == next_sent_id]
        sentence_tokens = [row["token_list"][i] for i in current_sent_positions]
        sentence_token_type_ids = [row["custom_token_type_ids"][i] for i in current_sent_positions]
        sentence_token_list.append(sentence_tokens)
        if synfeat==True: # set only the token_type_ids to 1 for the filtered utterance_token_list
            filtered_sentence_token_type_ids = []
            c = 0
            for i, value in enumerate(sentence_token_type_ids):
                if value == 1:
                    if c in row['synfeat_ids']:
                        filtered_sentence_token_type_ids.append(sentence_token_type_ids[i])
                    else:
                        filtered_sentence_token_type_ids.append(0)
                    c += 1 # start counting c once we hit 1 in the token_type_ids
                else:
                    filtered_sentence_token_type_ids.append(0)
            full_utterance_token_list = [sentence_tokens[i] for i, _ in enumerate(sentence_tokens) if
                                         sentence_token_type_ids[i] == 1]
            sanity_check = [sentence_tokens[i] for i, _ in enumerate(sentence_tokens) if filtered_sentence_token_type_ids[i] == 1]
            if sanity_check != row['utterance_token_list']:
                print("Mismatch:", row['utterance_token_list'], '/', sanity_check, '/', full_utterance_token_list, '/', row['synfeat_ids'])
            sentence_token_type_id_list.append(filtered_sentence_token_type_ids)
        else:
            sentence_token_type_id_list.append(sentence_token_type_ids)
    segment_df_full["sentence_token_list"] = sentence_token_list
    segment_df_full["sentence_token_type_ids"] = sentence_token_type_id_list

    return segment_df_full


def tokenize_no_alignment(examples, tokenizer):
    """
    This function tokenizes the input. The input consists only of the utterance in a given instance.

    :param examples: the untokenized transformers Dataset
    :param tokenizer: the defined transformers tokenizer to use for tokenization
    :return: tokenized_inputs: the tokenized tokens from the examples row containing the text, this input
    is ready for training or inference
    """

    tokenized_inputs = tokenizer(examples["utterance_token_list"],
                                 truncation=True,
                                 padding="max_length",
                                 max_length=256,
                                 is_split_into_words=True)

    return tokenized_inputs


def tokenize_and_align_token_type_ids(examples, tokenizer):
    """
    This function is similar to the classical tokenize_and_align_labels function.
    Since after tokenization, one word may get split into more than one subword(s),
    the tokenized sentence may not have the same length as the original sentence.
    This creates a length mismatch between our custom token type ids and the new
    tokenized sentence. We adapt the length of our custom token type ids to match
    the tokenized sentence, using the function word_ids().

    :param examples: the untokenized transformers Dataset
    :param tokenizer: the defined transformers tokenizer to use for tokenization
    :return: tokenized_inputs: the tokenized tokens from the examples row containing the text, this input
    is ready for training or inference
    """

    tokenized_inputs = tokenizer(examples["sentence_token_list"],
                                 truncation=True,
                                 # padding="longest",
                                 padding="max_length",
                                 max_length=256,
                                 is_split_into_words=True)

    aligned_token_type_ids = []

    for i, type_ids in enumerate(examples[f"sentence_token_type_ids"]):  # example: i: 91, type_ids: [1, 1, 0]
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # example: word_ids: [None, 0, 1, 2, None]
        previous_word_idx = None
        current_token_type_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                current_token_type_ids.append(0)
            else:
                current_token_type_ids.append(type_ids[word_idx])
            previous_word_idx = word_idx
        aligned_token_type_ids.append(current_token_type_ids)  # example: current_token_type_ids: [0, 1, 1, 0, 0]

    tokenized_inputs["token_type_ids"] = aligned_token_type_ids

    return tokenized_inputs


def preprocess_labels(tokenized_dataset, unique_labels):
    labels_batch = {k: tokenized_dataset[k] for k in tokenized_dataset.keys() if k in unique_labels}
    labels_matrix = np.zeros((len(tokenized_dataset["token_type_ids"]), len(unique_labels)))
    for idx, label in enumerate(unique_labels):
        labels_matrix[:, idx] = labels_batch[label]
    tokenized_dataset["labels"] = labels_matrix.tolist()

    return tokenized_dataset
