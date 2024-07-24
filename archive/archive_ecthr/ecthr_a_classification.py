from datasets import load_dataset

dataset = load_dataset("lex_glue", "ecthr_a")


# !python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token(XXXX)" insert token here

tokenizer = AutoTokenizer.from_pretrained("danielsaggau/simcse_longformer_ecthr_b", use_auth_token=True, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("danielsaggau/simcse_longformer_ecthr_b", num_labels=10, problem_type='multi_label_classification')

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) # fp16

import numpy as np

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
        
tokenized_data = dataset.map(preprocess_function, batched=True,remove_columns=['text'])

# cast label IDs to floats
import torch 
tokenized_data.set_format("torch")
tokenized_data = (tokenized_data
          .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels", "labels"))
 
 from transformers import EvalPrediction
from scipy.special import expit
from sklearn.metrics import f1_score

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
      
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="/slbert_ecthr_a_classsification",
    learning_rate=3e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=20,
    weight_decay=0.01,
    fp16=True,
    eval_steps=250,
    save_strategy="steps",
    evaluation_strategy="steps",
    logging_steps=250,
    logging_dir='./logs',  
    logging_first_step = True,
    logging_strategy = 'steps',
    #push_to_hub=True,
    metric_for_best_model="micro-f1",
    greater_is_better=True,
    load_best_model_at_end = True
)


from transformers import EarlyStoppingCallback
from transformers import Trainer
hist =Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=training_args,
    eval_dataset=tokenized_data['validation'],
    train_dataset=tokenized_data['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)])
hist.train()
