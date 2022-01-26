# thesis

# Potential Topics 

Explainability for Legal Judgment Prediction
Main topic: Explainability
I recently moved a bit to this direction for the very first time in this paper (https://aclanthology.org/2021.naacl-main.22/), where we examine rationale extraction on ECtHR cases, but recently two more datasets have been published (ILDC: https://aclanthology.org/2021.acl-long.313/, ContractNLI: https://aclanthology.org/2021.findings-emnlp.164/), which are annotated with both relevant labels and rationales (sentences that indicate the right label).
I guess a first goal here would be to consider recent advances in this topic (explainability / rationale extraction) and compare different methods across these datasets / tasks. We could even consider more classic methods, like LIME, that I skipped in this work.

An evaluation framework for probing abstract legal NLU
Main topic: Legal NLU /  Probing
The question here is "In which degree pre-trained language models (LMs), generic or legal-oriented, "understand" legal text? What kind of legal knowledge they hold? How can we estimate this?".
Here, we should first define a set of "probing" tasks (LM-based, or sentence retrieval) that somehow quantify this quality (legal language understanding). Of course, we’ll have to collaborate and learn a lot from people studying law. 
Then we could evaluate and analyse the performance of different models, generic ones (e.g., RoBERTa) or legal-oriented (Legal-BERT, CaseLaw-BERT).

Retrieval-enhanced Legal Document Classification
Main Topic: Multi-Document Classification 
The question here is "Is external knowledge -described in other documents- beneficial for legal document classification? How we could effectively model relevant documents and their relation with the target document (the one to be classified)?"
Recently, similar ideas have been examined in the context of LM pre-training (https://arxiv.org/abs/2112.04426), where retrieving "relevant" sentences seems to improve LM performance. But, in general, it seems natural to exploit relevant sources (documents) to resolve document classification tasks. For example, legal professionals (lawyers, judges) always consider the relevant legislation and case-law when they want to predict the outcome of a legal case, while we currently focus on models that only consider the facts of the case.
Based on this idea, we could examine how we could better (but also efficiently) encode relevant documents to improve the performance in legal judgment prediction.

A deeper dive (analysis) on what leads to group disparities in legal NLP tasks

Main Topic: Fairness / Robustness
We recently released a benchmark for fairness in legal NLP tasks (see the confidential pre-print attached). We showcased reasonable group performance disparities in several tasks w.r.t. to several attributes.  We examined several group-robust methods that are supposed to improve fairness by mitigating group performance disparities, but they seem to not work consistently or not at all in several cases. We failed to answer why this happens and what we could do about it. 
I think it would make sense to consider how different factors (distributional swifts, labelling inconsistency, or whatsoever) potentially affect the results. In many cases, maybe specific legal systems or courts are indeed unfair and treats several groups harsher (unequal), e.g., men vs. women, or eastern European countries vs. western, but maybe in other cases, it’s all (or in a large degree) about representation disparities (more or less data for some groups) or other reasons that affect the learning process.
We could probably examine how results and group-robust methods efficacy differ in different settings to approach and understand these issues.



# Literature Review 
 
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
