import torch 
from torch import nn
import logging
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EvalPrediction, HfArgumentParser, TrainingArguments, default_data_collator, set_seed, EarlyStoppingCallback, Trainer, TrainerCallback 
from datasets import load_metric, load_dataset
import numpy as np
import transformers
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from sklearn.metrics import f1_score
from scipy.special import expit
import glob
import shutil 
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.tokenization_utils_base import BatchEncoding
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')
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
        },
    )
    dataset_selection: str = field(
      default="ecthr_b",
      metadata={"help": "dataset selection argument which allows us to pick the correct dataset"}
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
    model_name: str = field(
      default = 'danielsaggau/bregman_ecthrb_k_10_ep1', metadata={'help': ' name of the model path'}
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    do_pooling: Optional[bool] = field(
        default=False,
        metadata={"help": "Do pooling"},
    )
    model_type:Optional[str] = field(
      default=max,
      metadata={"help": "determine the type of pooling and model loaded"}
    )
    freezing:Optional[bool] = field(
      default=False,
      metadata={"help": 'Freeze the layers apart from the Classification head (MLP)'}
    )
    subnetworks: Optional[bool] = field(
      default=False, 
      metadata={"help": 'number of subnetworks'}
    )
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
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
    
    
    
    label_list = list(range(10))
    num_labels = len(label_list)
    
 
      if data_args.dataset_selection=="ecthr_a":
        dataset = load_dataset("lex_glue", "ecthr_a")
        logger.info('load ecthr_a regular model')
      elif data_args.dataset_selection=='ecthr_b':
        dataset = load_dataset("lex_glue", "ecthr_b")
        logger.info('load ecthr_b regular model')
      tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=True, use_fast=True)
      model = AutoModelForSequenceClassification.from_pretrained(model_args.model_nam,
      num_labels=10, problem_type='multi_label_classification')


    # set padding to max length   
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        cases = []
        padding = "max_length"
        max_seq_length=4096
        for case in examples['text']:
            cases.append(f' {tokenizer.sep_token} '.join([fact for fact in case]))
        batch = tokenizer(cases, padding=padding, max_length=4096, truncation=True)
        # use global attention on CLS token
        global_attention_mask = np.zeros((len(cases),max_seq_length), dtype=np.int32)
        global_attention_mask[:, 0] = 1
        batch['global_attention_mask'] = list(global_attention_mask)
        batch["labels"] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in examples["labels"]]
        return batch

    tokenized_data = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['text'],
    desc="Tokenizing the entire dataset")

    tokenized_data.set_format("torch")
    tokenized_data = tokenized_data.map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"]).rename_column("float_labels", "labels")

    def compute_metrics(p: EvalPrediction):
        # Fix gold labels
        y_true = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
        y_true[:, :-1] = p.label_ids
        y_true[:, -1] = (np.sum(p.label_ids, axis=1) == 0).astype('int32')
        # Fix predictions
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = (expit(logits) > 0.5).astype('int32')
        y_pred = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
        y_pred[:, :-1] = preds
        y_pred[:, -1] = (np.sum(preds, axis=1) == 0).astype('int32')
        # Compute scores
        macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) # fp16

    if model_args.model_type =='mean':
        logger.info('model mean pooler loaded')
        class LongformerMeanPooler(nn.Module):
          def __init__(self, config):
             super().__init__()
             self.dense = nn.Linear(config.hidden_size, config.hidden_size)
             self.activation = nn.Tanh()

          def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
              mean_token_tensor = hidden_states.mean(dim=1)
              pooled_output = self.dense(mean_token_tensor)
              pooled_output = self.activation(pooled_output)
              return pooled_output
        model.longformer.pooler = LongformerMeanPooler(model.config)

    # freezing the body and only leaving the head 
    if model_args.freezing:
      for name, param in model.named_parameters():
        if name.startswith("longformer."): # choose whatever you like here
          param.requires_grad = False
          logger.info('Freeze All Parameters apart from the CLS head')

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
    wandb.init(project="IR_LDC",name="ECTHR_frozen")
    trainer.train()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_dataset=tokenized_data['test']
    trainer.evaluate(eval_dataset=eval_dataset)

if __name__ == "__main__":
    main()
