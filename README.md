# Retrieval-Augmented Generation in a Low Lexical Diversity Scenario
## Table of Contents  
- [Introduction](#introduction)  
- [Features](#features)  
- [Methodology](#methodology)  
- [Experiments](#experiments)   
- [Conclusion](#conclusion)   
- [References](#references)  
## Introduction
This repository contains the implementation of our study on Retrieval-Augmented Generation (RAG) in a low lexical diversity setting, specifically for classifying gas pipe damage descriptions based on their patchability.

Our work introduces a new lexical entropy metric to quantify textual diversity and proposes [+]EXPL, a data re-balancing technique that improves retrieval effectiveness in highly imbalanced datasets. The retrieval system is based on SBERT-NLI embeddings, while Mistral7B serves as the generative model for classification.

## Features
- Implements **Retrieval-Augmented Generation (RAG)** for classification.
- Introduces **Corpus Lexical Entropy** as a diversity metric.
- Uses **FAISS** for fast retrieval of similar cases.
- Employs **SBERT-NLI** embeddings for document representation.
- Integrates **Mistral7B** for classification.
- Proposes **[+]EXPL**, a downsampling strategy to mitigate class imbalance.

## Methodology

#### Mathematical Formulation-

The retrieval and classification processes follow these key steps:

1. **Encoding:** Documents and queries are embedded into a vector space using **SBERT-NLI**.
2. **Retrieval:** **FAISS** retrieves the top-**k** most similar cases using the dot product as a similarity metric.
3. **Generation:** **Mistral7B** processes the retrieved examples and classifies the query as either **patchable (YES)** or **non-patchable (NO)**.

The retrieval function selects **k** nearest documents:

$$
Re_{s,k,M} (q) = \arg\max_{N \subseteq M, |N| = k} \sum_{d \in N} s(e(q), e(d))
$$

where:
- \( e \) is the **encoder function** mapping documents to a vector space.
- \( s \) is the **similarity function** (dot product in this case).
- \( M \) is the **retrieval memory**.
- \( q \) is the **input query**.

#### Corpus Lexical Entropy-

We define **Corpus Lexical Entropy** as a measure of term diversity across the corpus:

$$
H(VD) = \sum_{t \in VD} H(t)
$$

#### Retriever-

The retrieval system consists of the following components:

- **Embedding function (e)**: **SBERT-NLI** is selected as the best embedding model based on retrieval effectiveness.
- **Similarity metric (s)**: The internal **dot product** is used to measure document similarity.
- **Memory selection (M)**: FAISS is leveraged for efficient retrieval.

#### Decoder-

The decoder (θ) is selected based on its ability to classify gas pipe damage. Mistral7B is chosen for its strong domain knowledge and few-shot learning capabilities.

## Experiments

#### Data-

The dataset consists of **11,904 cases** describing gas pipe damage conditions, with only **1.06%** being patchable. We apply **stratified sampling** to create training and test sets.

#### Evaluation Metrics-

- **F1-Macro score** to balance class imbalance effects.
- **Self-Consistency score (SC)** to measure model stability.

#### Results-

- **SBERT-NLI outperformed MPNet** in retrieval, demonstrating better recognition of subtle logical cues (e.g., negations, pressure levels).
- **Mistral7B achieved 0.68 F1-Macro in zero-shot classification**, improving to **0.87 with retrieval augmentation (RAG-[+]EXPL, k=9)**.
- **[+]EXPL significantly improved retrieval effectiveness**, removing overly similar negative cases near positive ones.

## Conclusion

This work demonstrates that **Retrieval-Augmented Generation (RAG) is effective in low-lexical-diversity settings**. Key findings include:

- **SBERT-NLI is the best retrieval encoder** for this dataset.
- **Mistral7B performs well with few-shot learning**, especially when **[+]EXPL improves retrieval quality**.
- **Fine-tuning was intentionally avoided** to maintain generalizability to human-written texts.

#### Future Directions-

- **Combining SBERT and MPNet** for hybrid retrieval.
- **Evaluating the model on real-world, multilingual datasets**.
- **Incorporating explanations in the output.**

## References

[1] Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, and Shengxin Zhu.  
_Unleashing the potential of prompt engineering in large language models: a comprehensive review._  
arXiv preprint arXiv:2310.14735, 2023.  

[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,  
Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.  
_Retrieval-augmented generation for knowledge-intensive NLP tasks._ 2021.  

[3] Nils Reimers and Iryna Gurevych.  
_Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks._  
Proceedings of EMNLP, 2019.  
