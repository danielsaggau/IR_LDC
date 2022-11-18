# loss function 
import torch
from torch import nn
from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor

class CustomBregmanLoss(nn.Module):
  '''
  The loss expects
  '''
  def __init__(self, model: SentenceTransformer, batch_size, temperature, sigma):
    '''
    param model
    param batch size
    param temperature:
    param sigma:  
    param cross entropy loss: 
    param mask:  
    Example: 
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    model = SentenceTransformer('distilbert-base-uncased')
    train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']), InputExample(texts=['Anchor 2', 'Positive 2'])]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
    train_loss = CustomBregmanLoss(model=model, batch_size=2,temperature=0.1, sigma=2) 
    model.fit([(train_dataloader, train_loss)], show_progress_bar=True)    
    '''  
    super(CustomBregmanLoss, self).__init__()
    self.model=model
    self.batch_size = batch_size
    self.temperature = temperature
    self.sigma = sigma
    self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")
    self.mask = self.mask_correlated_samples(batch_size)

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

  def forward(self, sentence_features:Iterable[Dict[str,Tensor]], labels:Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        N = 2 * self.batch_size
        features = torch.cat((embeddings_a, embeddings_b), dim=0)
        
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

        labels = torch.zeros(N, dtype=torch.long).to(device=features.device)
        scores = torch.cat((positives, negatives), dim=1)
        loss = self.cross_entropy_loss(scores, labels)
        loss /= N
        return loss#, num_max
