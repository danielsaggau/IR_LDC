import pandas as pd
import datasets
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import os

config = LongformerConfig()

config

LongformerConfig {
  "attention_probs_dropout_prob": 0.1,
  "attention_window": 512,
  "bos_token_id": 0,
  "eos_token_id": 2,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "longformer",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "sep_token_id": 2,
  "type_vocab_size": 2,
  "vocab_size": 30522
}

train_data, test_data = datasets.load_dataset('imdb', split =['train', 'test'], 
                                             cache_dir='/media/data_files/github/website_tutorials/data')


# load model and tokenizer and define length of the text sequence
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                           gradient_checkpointing=False,
                                                           attention_window = 512)
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)

model.config

LongformerConfig {
  "_name_or_path": "allenai/longformer-base-4096",
  "attention_mode": "longformer",
  "attention_probs_dropout_prob": 0.1,
  "attention_window": [
    512,
    512,
    512,
    512,
    512,
    512,
    512,
    512,
    512,
    512,
    512,
    512
  ],
  "bos_token_id": 0,
  "eos_token_id": 2,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "ignore_attention_mask": false,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 4098,
  "model_type": "longformer",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "sep_token_id": 2,
  "type_vocab_size": 1,
  "vocab_size": 50265
}

# define a function that will tokenize the model, and will return the relevant inputs for the model
def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding = 'max_length', truncation=True, max_length = 1024)

train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))

len(train_data['input_ids'][0])

# define the training arguments
training_args = TrainingArguments(
    output_dir = '/media/data_files/github/website_tutorials/results',
    num_train_epochs = 5,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 8,    
    per_device_eval_batch_size= 16,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps = 4,
    fp16 = True,
    logging_dir='/media/data_files/github/website_tutorials/logs',
    dataloader_num_workers = 0,
    run_name = 'longformer-classification-updated-rtx3090_paper_replication_2_warm'
)

# save the best model
trainer.save_model('/media/data_files/github/website_tutorials/results/paper_replication_lr_warmup200')

trainer.evaluate()


