# Thesis

# Mina Links: 

## Finding a divergence learning metric

https://proceedings.neurips.cc/paper/2020/file/c928d86ff00aeb89a39bd4a80e652a38-Paper.pdf

https://arxiv.org/pdf/1704.00454.pdf

Matthias Legal NLP General Info Links: 

https://aclanthology.org/2021.acl-long.213.pdf

https://arxiv.org/pdf/2010.00711.pdf

https://sites.ualberta.ca/~rabelo/COLIEE2022/

https://aclanthology.org/2021.nllp-1.13.pdf

https://nllpw.org/workshop/program/

https://aclanthology.org/2021.nllp-1.7.pdf

https://ostendorff.org/assets/pdf/ostendorff2021a.pdf

# General Interest Literature Review 
 
AdapterFusion: Non-Destructive Task Composition for Transfer Learning
https://aclanthology.org/2021.eacl-main.39.pdf

Parameter-Efficient Transfer Learning for NLP
http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf

EXT5: TOWARDS EXTREME MULTI-TASK SCALING FOR TRANSFER LEARNING
https://arxiv.org/pdf/2111.10952.pdf

mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
https://arxiv.org/pdf/2010.11934.pdf

Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark
https://arxiv.org/pdf/2110.00806.pdf

Lex Rosetta: Transfer of Predictive Models Across Languages, Jurisdictions, and Legal Domains
https://dl.acm.org/doi/pdf/10.1145/3462757.3466149

Dynamic Knowledge Distillation for Pre-trained Language Models
https://aclanthology.org/2021.emnlp-main.31.pdf

Towards Zero-Shot Knowledge Distillation for Natural Language Processing
https://aclanthology.org/2021.emnlp-main.526.pdf

Zero-Shot Cross-Lingual Transfer of Neural Machine Translation with Multilingual Pretrained Encoders
https://aclanthology.org/2021.emnlp-main.2.pdf

mT6: Multilingual Pretrained Text-to-Text Transformer with Translation Pairs
https://aclanthology.org/2021.emnlp-main.125.pdf

Cross-Attention is All You Need: Adapting Pretrained Transformers for Machine Translation
https://aclanthology.org/2021.emnlp-main.132.pdf

MDAPT: Multilingual Domain Adaptive Pretraining in a Single Model
https://arxiv.org/pdf/2109.06605.pdf

Visually Grounded Reasoning across Languages and Cultures
https://arxiv.org/pdf/2109.13238.pdf

Improving language models by retrieving from trillions of tokens
https://arxiv.org/pdf/2112.04426.pdf


# Ilias RQ: 

Retrieval-enhanced Legal Document Classification
Main Topic: Multi-Document Classification 
The question here is "Is external knowledge -described in other documents- beneficial for legal document classification? How we could effectively model relevant documents and their relation with the target document (the one to be classified)?"
Recently, similar ideas have been examined in the context of LM pre-training (https://arxiv.org/abs/2112.04426), where retrieving "relevant" sentences seems to improve LM performance. But, in general, it seems natural to exploit relevant sources (documents) to resolve document classification tasks. For example, legal professionals (lawyers, judges) always consider the relevant legislation and case-law when they want to predict the outcome of a legal case, while we currently focus on models that only consider the facts of the case.
Based on this idea, we could examine how we could better (but also efficiently) encode relevant documents to improve the performance in legal judgment prediction.
