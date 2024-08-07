import torch 
from torch import nn
import logging
import os
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
    TrainerCallback,
    Trainer
)
from datasets import load_metric
import numpy as np
import transformers
import logging
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


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

dataset = load_dataset("lex_glue", "scotus")

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.tokenization_utils_base import BatchEncoding

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('XXXX')
import wandb
logger = logging.getLogger(__name__)
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
        }
    )
    max_seq_length: Optional[int]= field(
      default=4096,
      metadata={
        "help":"Max sequence length"
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
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    do_pooling: Optional[bool] = field(
        default=False,
        metadata={"help": "Do pooling"},
    )
    compute_accuracy:Optional[bool] = field(
      default=False, 
      metadata={"help": "compute accuracy in addition to f1 micro and macro"}
    )
    model_name: str = field(
      default = 'danielsaggau/bregman_ecthrb_k_10_ep1', metadata={'help': ' name of the model path'}
    )
    model_type:Optional[str] = field(
      default=max,
      metadata={"help": "determine the type of pooling and model loaded"}
    )
    freezing:Optional[bool] = field(
      default=False,
      metadata={"help": 'Freeze the layers apart from the Classification head (MLP)'}
    )
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        #cudnn.deterministic = False
        #cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #set up device
    label_list = list(range(14)) #fix this in dynamic manner
    num_labels = len(label_list)

    if model_args.model_type =='mean':
      model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name,use_auth_token=True, num_labels=14).to(args.device)
      tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=True,use_fast=True)
      logger.info('load mean model')
    elif model_args.model_type =='cls':
      model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name, use_auth_token=True, num_labels=14).to(args.device)
      tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=True,use_fast=True)
      logger.info('load cls model')
    elif model_args.model_type =='max':
      model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name,use_auth_token=True, num_labels=14).to(args.device)
      tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=True,use_fast=True)
      logger.info('loading max model')


    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
      return tokenizer(examples["text"], truncation=True, padding=padding)
         
    tokenized_data = dataset.map(
      preprocess_function,
      batched=True,
      desc="tokenizing the entire dataset")
    
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
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        micro1 = metric1.compute(predictions=predictions, references=labels, average="micro")["f1"]
        macro1 = metric1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        return { "f1-micro": micro1, "f1-macro": macro1} 

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) # fp16
    
    if model_args.model_type =='mean':
        class LongformerMeanPooler(nn.Module):
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
        model.longformer.pooler = LongformerMeanPooler(model.config).to(args.device)
        print('model mean pooler loaded')
    
    elif model_args.model_type =='max':
      logger.info('Instantiate max pooling')
      class LongformerMeanPooler(nn.Module):
        def __init__(self, config, pooling='max'):
          super().__init__()
          self.dense = nn.Linear(config.hidden_size, config.hidden_size)
          self.pooling = pooling
          self.activation = nn.Tanh()
          self.max_sentence_length = 512

        def forward(self, hidden_states):
            pooled_output = torch.max(hidden_states, dim=1)[0]
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            return pooled_output
      model.longformer.pooler = LongformerMeanPooler(model.config).to(args.device)
      
    elif model_args.model_type=="cls":
      class LongformerCLSPooler(nn.Module):
        def __init__(self, config):
          super().__init__()
          self.dense = nn.Linear(config.hidden_size, config.hidden_size)
          self.activation = nn.Tanh()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

          mean_token_tensor = hidden_states[:, 0]
          pooled_output = self.dense(mean_token_tensor)
          pooled_output = self.activation(pooled_output)
          return pooled_output
      model.longformer.pooler = LongformerCLSPooler(model.config).to(args.device)
      logger.info('model cls pooler loaded')

    # freezing the body and only leaving the head 
    if model_args.freezing: 
        for name, param in model.named_parameters():
            if name.startswith("longformer."): # choose whatever you like here
                param.requires_grad = False
            logger.info('Freeze All Parameters apart from the CLS head')


    model.to(args.device)

    trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=training_args,
    eval_dataset=tokenized_data['validation'],
    train_dataset=tokenized_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
      )
    wandb.init(project="IR_LDC",name="bregman_scotus_s3_test")
    trainer.train()
    trainer.save_state()

    eval_dataset=tokenized_data['test']
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
if __name__ == "__main__":
    main()
