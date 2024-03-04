import torch
import numpy as np
import logger
import logging
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler  
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup 
from transformers import set_seed
from datasets import load_dataset  
from sklearn.metrics import classification_report
import helpers
from time import gmtime, strftime
import random
import os, sys
import configparser



###############################
# Read config and set seed
config_parser = configparser.ConfigParser()
config_parser.read(sys.argv[1])

logging.basicConfig(level=logging.INFO)


###############################
### Data paths:
### 
testfile  = config_parser['DATA']['filepath_test'] 
resfolder = config_parser['DATA']['result_folder']

if not os.path.isdir(resfolder): os.mkdir(resfolder)

###############################
### Parameter settings:
### 
BATCH_SIZE = int(config_parser['PARAM']['batch_size'])

###############################
### Model settings:
model_name = config_parser['MODEL']['bert_model']
bert_tokenizer = config_parser['MODEL']['bert_tokenizer']
model_abbr = config_parser['MODEL']['model_abbr'] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####
# Load data and extract label indices
data = load_dataset('json', data_files={'test': testfile })
labels = ["[PAD]", "[UNK]", "B", "I", "O"]
  
    
label2index, index2label = {}, {}
for i, item in enumerate(labels):
    label2index[item] = i
    index2label[i] = item


bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2index)).to("cuda")
    
    

def tokenize_and_align_labels(examples):
    tokenized_inputs = bert_tokenizer(examples["words"],
                                      truncation=True,
                                      padding='max_length',
                                      max_length=512,
                                      is_split_into_words=True)
    labels = []; predicates = []

    for idx, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = [] 
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label2index[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels 
    return tokenized_inputs


def encode_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['words', 'tags'])


def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2label[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2label[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list



#############
### Load data
data_encoded = encode_dataset(data)

te_input_ids, te_attention_masks, te_label_ids, te_seq_lengths = helpers.load_input(data_encoded["test"])
test_dataset  = TensorDataset(te_input_ids, te_attention_masks, te_label_ids, te_seq_lengths)




######################
### create data loader
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=0)


###################
### Test loop   ###
 
def model_eval(model, test_dataloader, dic):

    model.eval()
    total_sents = 0

    for batch in test_dataloader:
        # Add batch to GPU
        # Unpack the inputs from our dataloader
        t_input_ids, t_input_masks, t_labels, t_lengths = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(t_input_ids, attention_mask=t_input_masks)

        logits = outputs[0]
        class_probabilities = torch.softmax(logits, dim=-1)

        # Move class_probabilities and labels to CPU
        class_probabilities = class_probabilities.detach().cpu().numpy()
        argmax_indices = np.argmax(class_probabilities, axis=-1)

        label_ids = t_labels.to('cpu').numpy()
        #token_ids = t_token_type_ids.to('cpu').numpy()
        seq_lengths = t_lengths.to('cpu').numpy()


        for ix in range(len(label_ids)):
            total_sents += 1

            # Store predictions and true labels
            pred_labels = [index2label[argmax_indices[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]
            gold_labels = [] #, token_labels = [], []
            for g in range(len(label_ids[ix])):
                if label_ids[ix][g] != -100: 
                    gold_labels.append(index2label[label_ids[ix][g]])
                    #token_labels.append(token_ids[ix][g])

            if len(pred_labels) != len(gold_labels):
                logging.info("Predictions not as long as gold: %s", total_sents)

            text = bert_tokenizer.convert_ids_to_tokens(t_input_ids[ix], skip_special_tokens=False)
            clean_text = []
            for i in range(1, len(text)):
                if label_ids[ix][i] == -100:
                    clean_text[-1] += text[i].replace('##', '').replace('[SEP]', '').replace('[PAD]', '')
                else:
                    clean_text.append(text[i])
            if len(clean_text) != len(pred_labels) or len(clean_text) != len(gold_labels):
                logging.info("ERROR: %s %s %s", len(clean_text), len(gold_labels), len(pred_labels))
                logging.info("%s", clean_text)
                logging.info("%s", gold_labels)
            dic["words"].append(clean_text)
            dic["gold"].append(gold_labels)
            dic["pred"].append(pred_labels)

    return dic


#############################
### Print results to file ###
def print_results(results_str, resfolder, resfile):
    resfile = resfolder + '/' + resfile 
    with open(resfile, "w") as out:
        out.write(results_str + "\n")
    return


#############################
### Print predictions to file ###
def print_predictions(dic, predfolder, predfile):
    predfile = predfolder + '/' + predfile 
    gold_labels, pred_labels = [], []
    with open(predfile, "w") as out:
        for i in range(len(dic["words"])):
            for j in range(len(dic["words"][i])): 
                out.write(dic["words"][i][j] + "\t" + dic["gold"][i][j] + "\t" + dic["pred"][i][j] + "\n")
                gold_labels.append(dic["gold"][i][j])
                pred_labels.append(dic["pred"][i][j])
            out.write("\n")    
    clf_report = classification_report(gold_labels, pred_labels)
    return clf_report





def main():

    # evaluate baseline model on train data (de)
    test_dic = { "words":[], "gold":[], "pred":[] }

    test_dic = model_eval(model, test_dataloader, test_dic)
    test_predfile = 'predictions.txt'
    test_resfile = 'results.txt'
    logging.info("print test results to %s in folder %s", test_resfile, resfolder)
    results_test = print_predictions(test_dic, resfolder, test_predfile)
    print_results(results_test, resfolder, test_resfile)



if __name__ == "__main__":
    main()
