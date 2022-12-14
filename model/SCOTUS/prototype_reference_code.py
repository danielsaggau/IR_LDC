import torch
import torch.nn as nn
from datasets import Dataset
from transformers import FeatureExtractor

class PrototypeNet(nn.Module):
    def __init__(self, encoder, feature_extractor):
        super(PrototypeNet, self).__init__()
        self.encoder = encoder
        self.feature_extractor = feature_extractor

    def forward(self, x, y):
        # Compute the hidden representations of the input text
        h = self.encoder(x)

        # Extract the features for the input text using the HuggingFace feature extractor
        features = self.feature_extractor(x)

        # Compute the prototype representations for each class
        prototypes = self.get_prototypes(h, y)

        # Compute the gold labels as the cosine similarity between
        # the features and the prototype representations for each class
        gold_labels = self.get_gold_labels(features, prototypes)

        # Compute the few-shot learning loss
        loss = self.few_shot_loss(h, gold_labels)
        return loss

    def get_prototypes(self, h, y):
        # Initialize a dictionary to store the prototype representations for each class
        prototypes = {}

        # Loop over the classes in the dataset
        for label in dataset.labels:
            # Get all hidden representations with the current label
            h_with_label = h[y == label]

            # Compute the prototype representation for the current class
            prototype = torch.mean(h_with_label, dim=0)

            # Add the prototype representation to the dictionary
            prototypes[label] = prototype

        return prototypes

    def get_gold_labels(self, features, prototypes):
        #
