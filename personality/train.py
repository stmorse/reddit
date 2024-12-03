import time
import json
import pickle
import re
import os

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset, Dataset
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

from sklearn.metrics import accuracy_score, f1_score


BASE_PATH = '/sciclone/geograd/stmorse/reddit/'
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_LABELS = 16 
TTS = 1000000       # train-test-split


# helper function for preprocessing (tokenizing)
# def sanitize_text(text):
#     # Remove escape characters like \n, \t, etc.
#     text = text.replace("\n", " ").replace("\t", " ")
#     # Optionally remove non-printable characters
#     text = re.sub(r'[^\x20-\x7E]', '', text)
#     return text



# helper function for training/evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def main():
    # check GPU
    print(f'CUDA available? {torch.cuda.is_available()}')

    # don't want to fork process after tokenization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # don't want to use WANDB
    os.environ["WANDB_DISABLED"] = "true"

    # get start time
    t0 = time.time()

    # load data
    print(f'Loading data ... {time.time()-t0:.3f}')
    df_clean = pd.read_csv(f'{BASE_PATH}mbti_data/mbti_clean_v2.csv')
    # df_mini_train = pd.read_csv(f'{BASE_PATH}mbti_data/little_clean_v2_train.csv')
    # df_mini_test = pd.read_csv(f'{BASE_PATH}mbti_data/little_clean_v2_test.csv')

    # train_ds = Dataset.from_pandas(df_mini_train, split='train')
    # test_ds = Dataset.from_pandas(df_mini_test, split='test')
    train_ds = Dataset.from_pandas(df_clean.iloc[:TTS], split='train')
    test_ds = Dataset.from_pandas(df_clean.iloc[TTS:], split='test')
    # train_ds = load_dataset('csv', data_files={'train': f'{BASE_PATH}mbti_data/little_clean_v2_train.csv'})
    # test_ds = load_dataset('csv', data_files={'test': f'{BASE_PATH}mbti_data/little_clean_v2_test.csv'})

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        # examples["sentence"] = [sanitize_text(sentence) for sentence in examples["sentence"]]
        return tokenizer(examples["sentence"], truncation=True, padding=True)

    # tokenize train/test
    print(f'Tokenizing data ... {time.time()-t0:.3f}')
    tokenized_train_dataset = train_ds.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_ds.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS)

    # define training arguments
    training_args = TrainingArguments(
        output_dir="./results",         # Directory to save checkpoints
        eval_strategy='no',
        num_train_epochs=3.0,
        logging_dir="./logs",
        logging_strategy='steps',
        logging_steps=500,
        save_strategy='epoch',
        report_to='none',
    )

    # initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f'Training ... {time.time()-t0:.3f}')
    trainer.train()

    print(f'Evaluation ... {time.time()-t0:.3f}')
    trainer.evaluate()

    print(f'Saving model ... ')
    trainer.save_model('./final_model')

    print(f'Complete. {time.time()-t0:.3f}')


if __name__=="__main__":
    main()