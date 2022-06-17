import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers import LongformerSelfAttention
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LegalBertSelfAttention(LongformerSelfAttention) :
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask = None,
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            output_attentions =False,
    ):
        return super().forward(hidden_states, attention_mask = attention_mask, output_attentions= output_attentions)

    class LegalBertForMaskedLM: