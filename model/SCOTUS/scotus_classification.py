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

from datasets import load_dataset
dataset = load_dataset("lex_glue", "scotus")

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

logging.info('Connect to huggingface')

tokenizer = AutoTokenizer.from_pretrained('danielsaggau/longformer_simcse_scotus', use_auth_token=True,use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('danielsaggau/longformer_simcse_scotus',use_auth_token=True, num_labels=14)

logger = logging.getLogger(__name__)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_data = dataset.map(preprocess_function, batched=True)

logging.info('Tokenize the data')

def compute_metrics(eval_pred):
    metric1 = load_metric("f1")
    accuracy = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    micro1 = metric1.compute(predictions=predictions, references=labels, average="micro")["f1"]
    macro1 = metric1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = accuracy.compute(references=labels, predictions=predictions)['accuracy']
    return { "f1-micro": micro1, "f1-macro": macro1, "accuracy": accuracy}


training_args = TrainingArguments(
    output_dir="/scotus_experiments_MEAN_POOL_1",
    learning_rate=3e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    push_to_hub=True,
    fp16=True,
    warmup_ratio=3e-5,
    gradient_accumulation_steps=1,
    metric_for_best_model="f1-micro",
    greater_is_better=True,
#    lr_scheduler_type='cosine',
    load_best_model_at_end = True
)

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) # fp16

model = AutoModelForSequenceClassification.from_pretrained('danielsaggau/longformer_simcse_scotus',use_auth_token=True, num_labels=14)

import torch
from torch import nn
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

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=training_args,
    eval_dataset=tokenized_data['test'],
    train_dataset=tokenized_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)])
trainer.train()

eval_dataset=tokenized_data['validation']
trainer.evaluate(eval_dataset=eval_dataset)
#logger.warning(
 #       f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
 #       + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
 #   )
