from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
dataset=load_dataset("lex_glue",'scotus')

train_0 = dataset.filter(lambda example: example['label'] == 0)['train']# rows 1011
train_1 = dataset.filter(lambda example: example['label'] == 1)['train']# rows 811
train_2 = dataset.filter(lambda example: example['label'] == 2)['train']# rows 423
train_3 = dataset.filter(lambda example: example['label'] == 3)['train']# rows 193
train_4 = dataset.filter(lambda example: example['label'] == 4)['train']# rows 45
train_5 = dataset.filter(lambda example: example['label'] == 5)['train']# rows 35
train_6 = dataset.filter(lambda example: example['label'] == 6)['train']# rows 255
train_7 = dataset.filter(lambda example: example['label'] == 7)['train'] # rows 1043
train_8 = dataset.filter(lambda example: example['label'] == 8)['train'] # rows 717
train_9 = dataset.filter(lambda example: example['label'] == 9)['train']# rows 191
train_10 = dataset.filter(lambda example: example['label'] == 10)['train']# rows 53
train_11 = dataset.filter(lambda example: example['label'] == 11)['train']# rows 220
train_12 = dataset.filter(lambda example: example['label'] == 12)['train']# rows 2
train_13 = dataset.filter(lambda example: example['label'] == 13)['train'] # # rows 0

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

pipe = pipeline('feature-extraction', 
                      'danielsaggau/longformer_simcse_scotus', 
                      use_auth_token='XXXX')
from tqdm.auto import tqdm

for t0 in tqdm(pipe(KeyDataset(train_0, "text"), batch_size=32, truncation=True)):
    print(t0)
torch.save(t0, 'file_t0.pt')

for t1 in tqdm(pipe(KeyDataset(train_1, "text"), batch_size=32, truncation=True)):
    print(t1)
torch.save(t1, 'file_t1.pt')

for t2 in tqdm(pipe(KeyDataset(train_2, "text"), batch_size=32, truncation=True)):
    print(t2)
torch.save(t2, 'ile_t2.pt')

for t3 in tqdm(pipe(KeyDataset(train_3, "text"), batch_size=32, truncation=True)):
    print(t3)
torch.save(t3, 'file_t3.pt')

for t4 in tqdm(pipe(KeyDataset(train_4, "text"), batch_size=32, truncation=True)):
    print(t4)
torch.save(t4, 'file_t4.pt')

for t8 in tqdm(pipe(KeyDataset(train_8, "text"), batch_size=32, truncation=True)):
    print(t8)
torch.save(t8, 'file_t8.pt')

for t9 in tqdm(pipe(KeyDataset(train_9, "text"), batch_size=32, truncation=True)):
    print(t9)
torch.save(t9, 'file_t9.pt')

for t10 in tqdm(pipe(KeyDataset(train_10, "text"), batch_size=32, truncation=True)):
    print(t10)
torch.save(t10, 'file_t10.pt')

for t11 in tqdm(pipe(KeyDataset(train_11, "text"), batch_size=32, truncation=True)):
    print(t11)
torch.save(t11, 'file_t11.pt')
