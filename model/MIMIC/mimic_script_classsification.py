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
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datasets import load_dataset
from sklearn.metrics import f1_score
from scipy.special import expit
import glob
import shutil
from datasets import load_dataset
import torch
from transformers import Trainer
from scipy.special import expit
from sklearn.metrics import f1_score
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import shutil
#shutil.unpack_archive(filename, extract_dir)

dataset = load_dataset("/content/IR_LDC/model/MIMIC/mimic-dataset.py")
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
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
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
      default = 'content/IR_LDC/models/bio-longformer', metadata={'help': ' name of the model path'}
    )
    freezing: Optional[bool] = field(
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
  #  logger.warning(
  #      f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
  #      + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
  #  )
    logger.info(f"Training/evaluation parameters {training_args}")
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        #cudnn.deterministic = False
        #cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
      
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name,use_auth_token=True, num_labels=14).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=True,use_fast=True)
    logger.info('load mean model')  

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) # fp16
    train_dataset=dataset['train']
    num_labels = train_dataset.features['labels'].feature.num_classes
    label_ids = train_dataset.features['labels'].feature.names
    label_names = label_ids
    label_list = list(range(num_labels))
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
    
    def preprocess_function(examples):
        # Tokenize the texts
        batch = tokenizer(
            examples["text"],
            padding=padding,
            max_length=512,
            truncation=True)
        
        batch = tokenizer.pad(
            batch,
            padding='max_length',
            max_length=512,
            pad_to_multiple_of=8,
        )
        batch["label_ids"] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in examples["labels"]]
        return batch

    tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=['labels'])

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = (expit(logits) > 0.5).astype(int)
        label_ids = (p.label_ids > 0.5).astype(int)
        macro_f1 = f1_score(y_true=label_ids, y_pred=preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=label_ids, y_pred=preds, average='micro', zero_division=0)
        return {'macro_f1': macro_f1, 'micro_f1': micro_f1}


    training_args = TrainingArguments(
    output_dir="/biobert_mimic_classification_bregman_Tuned",
    learning_rate=1e-3,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    push_to_hub=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    load_best_model_at_end = True,
    report_to="wandb",
    run_name="mimic_bregman_biobert")

#Bert pooling

    class BertMeanPooler(nn.Module):
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
    model.bert.pooler = BertMeanPooler(model.config)
    print('model mean pooler loaded')
    if model_args.freezing: 
      for param in model.bert.parameters():
          param.requires_grad = False

    tune = Trainer(
      model=model,
      compute_metrics=compute_metrics,
      args=training_args,
      eval_dataset=tokenized_data['validation'],
      train_dataset=tokenized_data["train"],
      tokenizer=tokenizer,
      data_collator=data_collator,    
      callbacks = [EarlyStoppingCallback(early_stopping_patience=5)])
    tune.train()

    eval_dataset=tokenized_data['test']
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()



















