import torch 
from torch import nn
from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer, LoggingHandler, losses, InputExample,  models
from torch.utils.data import DataLoader
import model.loss.cos_sim


class AddProjection(nn.Module):
   def __init__(self, model: SentenceTransformer, sentence_embedding_dimension:int ,mlp_dim=512,embedding_size=5120):
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
        self.projection = AddProjection(model, self, mlp_dim=feat_dim)
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

     
