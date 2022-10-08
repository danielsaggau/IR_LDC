
class BregmanLoss(nn.Module):
    """
    The Bregman loss. Expect as Input 

    """ 

    def __init__(self, batch_size, model: SentenceTransformer,sigma, temperature, margin: float = 0.5, size_average:bool = True):
        super(BregmanLoss, self).__init__()
        self.margin = margin
        self.model = model
        self.temperature = temperature
        self.sigma = sigma
        self.batch_size = batch_size
        self.size_average = size_average
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def b_sim(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):): # double check the issue here 
        
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        
        mm = torch.max(reps, dim=1) # was features instead of reps 
        indx_max_features = mm[1]
        max_features = mm[0].reshape(-1, 1)
        
        # Compute the number of active subnets in one batch
        eye = torch.eye(features.shape[1])
        one = eye[indx_max_features]
        num_max = torch.sum(one, dim=0)
        
        dist_matrix = max_features - features[:, indx_max_features]
        
        case = 2
        if case == 0:
            m2 = torch.divide(dist_matrix, torch.max(dist_matrix))
            sim_matrix = torch.divide(torch.tensor([1]).to(features.device), m2 + 1)
            
        if case == 1:
            gamma = torch.tensor([1]).to(features.device)
            sim_matrix = torch.exp(torch.mul(-dist_matrix, gamma))
            
        if case == 2:
            sigma = torch.tensor([self.sigma]).to(features.device)
            sig2 = 2 * torch.pow(sigma, 2)
            sim_matrix = torch.exp(torch.div(-dist_matrix, sig2))
        
        if case == 3:
            sim_matrix = 1 - dist_matrix
            
        return sim_matrix, num_max

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        N = 2 * self.batch_size
        ###################################################
        ### Computing Similarity Matrix ###################
        sim_matrix, num_max = self.b_sim(features) # todo features vs reps
        sim_matrix = sim_matrix / self.temperature
        ##################################################

        pos_ab = torch.diag(sim_matrix, self.batch_size)
        pos_ba = torch.diag(sim_matrix, -self.batch_size)

        positives = torch.cat((pos_ab, pos_ba), dim=0).reshape(N, 1)
        negatives = sim_matrix[self.mask].reshape(N, -1)

        labels = torch.zeros(N, dtype=torch.long).to(features.device)
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss, num_max
