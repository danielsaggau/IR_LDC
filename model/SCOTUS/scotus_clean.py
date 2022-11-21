import torch 
from torch import nn
import logging
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    Trainer
)
from transformers import TrainerCallback 
from datasets import load_metric
import numpy as np

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datasets import load_dataset
from sklearn.metrics import f1_score
from scipy.special import expit
import glob
import shutil

from datasets import load_dataset
dataset = load_dataset("lex_glue", "scotus")

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.tokenization_utils_base import BatchEncoding

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

logging.info('Connect to huggingface')

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    #model_name: Optional[str] = field(
    #    default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    #)
    OUTPUT_PATH: str = field(
        default='output', metadata={"help": "Path to output repository locally"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    do_pooling: Optional[bool] = field(
        default=False,
        metadata={"help": "Do pooling"},
    )
    compute_accuracy:Optional[bool] = field(
      default=False, 
      metadata={"help": "compute accuracy in addition to f1 micro and macro"}
    )
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained('danielsaggau/longformer_simcse_scotus', use_auth_token=True,use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained('danielsaggau/longformer_simcse_scotus',use_auth_token=True, num_labels=14)

    logger = logging.getLogger(__name__)

    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
      return tokenizer(examples["text"], truncation=True, padding=padding)


    tokenized_data = dataset.map(preprocess_function, batched=True)

    logging.info('Tokenize the data')



    if model_args.compute_accuracy: 
      def compute_metrics(eval_pred):
        metric1 = load_metric("f1")
        accuracy = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        micro1 = metric1.compute(predictions=predictions, references=labels, average="micro")["f1"]
        macro1 = metric1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        accuracy = accuracy.compute(references=labels, predictions=predictions)['accuracy']
        return { "f1-micro": micro1, "f1-macro": macro1, "accuracy": accuracy}
    else: 
      def compute_metrics(eval_pred):
        metric1 = load_metric("f1")
        accuracy = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        micro1 = metric1.compute(predictions=predictions, references=labels, average="micro")["f1"]
        macro1 = metric1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        accuracy = accuracy.compute(references=labels, predictions=predictions)['accuracy']
        return { "f1-micro": micro1, "f1-macro": macro1, "accuracy": accuracy} 

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) # fp16
    model = AutoModelForSequenceClassification.from_pretrained('danielsaggau/longformer_simcse_scotus',use_auth_token=True, num_labels=14)

    if model_args.do_pooling:
        class CustomLongformerPooler(nn.Module):
          def __init__(self, config):
             super().__init__()
             self.dense = nn.Linear(config.hidden_size, config.hidden_size)
             self.activation = nn.Tanh()

          def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
              mean_token_tensor = hidden_states.mean(dim=1)
              pooled_output = self.dense(mean_token_tensor)
              pooled_output = self.activation(pooled_output)
              return pooled_output

        model.longformer.pooler = CustomLongformerPooler(model.config)
        print('model pooler loaded')
    trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=training_args,
    eval_dataset=tokenized_data['test'],
    train_dataset=tokenized_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
      )
    trainer.train()

    eval_dataset=tokenized_data['validation']
    trainer.evaluate(eval_dataset=eval_dataset)

if __name__ == "__main__":
    main()
