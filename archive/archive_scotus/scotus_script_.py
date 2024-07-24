!pip install huggingface_hub
#!python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('XXXX')"
from datasets import load_dataset
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
import logging
from datetime import datetime
import gzip
import sys
import tqdm

class DataTrainingArguments:
  '''
  arguments todo
  '''




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


# data processing

dataset = load_dataset("lex_glue", "scotus")

with open('your_file.txt', 'w') as f:
    for line in dataset['train']['text']:
        f.write(f"{line}\n")

# Training parameters
access="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"
model_name = 'danielsaggau/legal_long_bert'
train_batch_size = 6
max_seq_length = 4096
num_epochs = 8

#Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print("Run this script with: python {} path/to/sentences.txt".format(sys.argv[0]))
    exit()

#filepath = sys.argv[1]
filepath = "/content/IR_LDC/your_file.txt"

# Save path to store our model
output_name = ''
if len(sys.argv) >= 3:
    output_name = "-"+sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = 'output/train_simcse{}-{}'.format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

################# Read the train corpus  #################


train_samples = []
with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, desc='Read file'):
        line = line.strip()
        if len(line) >= 10:
            train_samples.append(InputExample(texts=[line, line]))

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up


from sentence_transformers import SentenceTransformer, models
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls') # remove this block to do mean pooling
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=8,
          warmup_steps=warmup_steps,
          steps_per_epoch=5000,
          callback="epoch",
          output_path='/content/drive/MyDrive/SIMCSE_SCOTUS_CLS',
          optimizer_params={'lr': 3e-5},
          checkpoint_path='/content/drive/MyDrive/SIMCSE_SCOTUS_CLS/output',
          show_progress_bar=True,
          checkpoint_save_steps=10000,
          save_best_model=True,
          use_amp=True  # Set to True, if your GPU supports FP16 cores
          )

# custom fit function s.t. combine loss function 





