from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, LoggingHandler, SentenceTransformer, InputExample
import logging
from datetime import datetime
import gzip
import sys
import tqdm

from datasets import load_dataset
dataset = load_dataset("lex_glue", "scotus")

with open('your_file.txt', 'w') as f:
    for line in dataset['train']['text']:
        f.write(f"{line}\n")

# Training parameters
access="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"
model_name = 'danielsaggau/legal_long_bert'
train_batch_size = 2
max_seq_length = 4096

#filepath = sys.argv[1]
filepath = "your_file.txt"

# Save path to store our model
output_name = ''
if len(sys.argv) >= 3:
    output_name = "-"+sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = 'output/train_simcse{}-{}'.format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
model = SentenceTransformer(model_name, use_auth_token=access)

################# Read the train corpus  #################
train_samples = []
with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, desc='Read file'):
        line = line.strip()
        if len(line) >= 10:
            train_samples.append(InputExample(texts=[line, line]))

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)



!python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')" 


import torch 
from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample,  models
from torch.utils.data import DataLoader

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
     


class AddProjection(nn.Module):
   def __init__(self, model: SentenceTransformer, mlp_dim=512,embedding_size=512*5): #removed sentence_embedding_dimension
       super(AddProjection, self).__init__()
       self.model = SentenceTransformer('danielsaggau/legal_long_bert')
       embedding_size = embedding_size
       mlp_dim =  self.model.get_sentence_embedding_dimension() 
       #self.model.fc = nn.Identity()
       self.projection = nn.Sequential(
           nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
           nn.BatchNorm1d(mlp_dim),
           nn.ReLU(),
           nn.Linear(in_features=mlp_dim, out_features=embedding_size),
           nn.BatchNorm1d(embedding_size),
       )

   def forward(self, a: Tensor):
       if not isinstance(a, torch.Tensor):
          a = torch.tensor(a)
       if len(a.shape) == 1:
          a = a.unsqueeze(0)
       a = self.projection(a)
       return a

class BregmanRankingLoss(nn.Module):
  '''

  '''
  def __init__(self, model: SentenceTransformer, sigma, temperature, batch_size, lambda1, lambda2 ,feat_dim=512, scale: float = 20.0, similarity_fct = cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(BregmanRankingLoss, self).__init__()
        self.model = model
        self.projection = AddProjection(self,model)
        self.sigma = sigma
        self.temperature = temperature
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(batch_size)
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


  def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.long)#, dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

  def b_sim(self, features):
        mm = torch.max(features, dim=1)
        indx_max_features = mm[1]
        max_features = mm[0].reshape(-1, 1)
        # Compute the number of active subnets in one batch
        eye = torch.eye(features.shape[1])
        one = eye[indx_max_features]
        num_max = torch.sum(one, dim=0)
        dist_matrix = max_features - features[:, indx_max_features]
        sigma = torch.tensor([self.sigma]).to(features.device)
        sig2 = 2 * torch.pow(sigma, 2)
        sim_matrix = torch.exp(torch.div(-dist_matrix, sig2))

        return sim_matrix, num_max

  def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features] # get output main model
        embeddings_a = reps[0] 
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        rloss = self.cross_entropy_loss(scores, labels)
        
        # bregman part 

        N = 2 * self.batch_size
        z1 = self.projection(embeddings_a)
        z2 = self.projection(embeddings_b)
        features = torch.cat((z1, z2), dim=0)

        ###################################################
        ### Computing Similarity Matrix ###################
        sim_matrix, num_max = self.b_sim(features)
        sim_matrix = sim_matrix / self.temperature
        ###################################################
        #sim_matrix = self.similarity_f(out.unsqueeze(1), out.unsqueeze(0)) / self.temperature

        pos_ab = torch.diag(sim_matrix, self.batch_size)
        pos_ba = torch.diag(sim_matrix, -self.batch_size)

        positives = torch.cat((pos_ab, pos_ba), dim=0).reshape(N, 1)
        negatives = sim_matrix[self.mask].reshape(N, -1)

        blabel = torch.zeros(N, dtype=torch.long).to(device=features.device)
        bscores = torch.cat((positives, negatives), dim=1)
        bloss = self.criterion(bscores, blabel)
        bloss /= N
        loss = self.lambda1* bloss + self.lambda2 * rloss 
        return loss




train_loss = BregmanRankingLoss(model=model, batch_size=2,temperature=0.1, sigma=2 ,lambda1=1, lambda2=2) 
num_epochs=10
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          steps_per_epoch=5000,
          callback="epoch",
          output_path='bregman_scotus_k5_ep10',
          optimizer_params={'lr': 3e-5},
          checkpoint_path='bregman_scotus_k5_ep10',
          show_progress_bar=True,
          checkpoint_save_steps=10000,
          save_best_model=True,
          use_amp=True  # Set to True, if your GPU supports FP16 cores
          )
