class BregmannTripletLoss(nn.Module):
      """ The Bregman loss should take a triplet (anchor, negative, positive) computing the loss for all valid triplets
      Arguments: 
      :param model: 
      :param embedding:
      negative
      Returns:
      Example:: 
      """
      def __init__(self, batch_size, temperature, sigma):
        super(BregmanLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.sigma = sigma
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
      def pairwise_divergence(embeddings, squared =False):
          max_output = torch.argmax(embeddings,1, output_type=torch.Tensor.int32)
          one_to_n = torch.range(torch.Tensor.int(embedding)[0], output_type=torch.Tensor.int32)
          max_indices = torch.transpose(torch.stack([one_to_n, max_out]))
          max_val = 
          # max_values = tf.gather_nd(embed, max_indices)
          # torch.gather(input = embeddings, index = max_indices, dim =)?
          max_val_repeated =
          repeated_max_out =
          repeated_one_to_n = 

      def forward(self, out_a, out_b): 
        similarity_matrix = ()
        features = torch.cat((out_a, out_b), dim=0)
        
#computing triplet 
        anchor_positive = torch.diag()
        anchor_negative = torch.diag()

      return div_matrix
