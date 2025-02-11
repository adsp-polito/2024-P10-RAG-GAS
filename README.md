# Retrieval-Augmented Generation in a Low Lexical Diversity Scenario
## Table of Contents
Introduction
Features
Methodology
Mathematical Formulation
Corpus Lexical Entropy
Retriever
Decoder
Experiments
Data
Evaluation Metrics
Results
Conclusion
References

## Introduction
This repository contains the implementation of our study on Retrieval-Augmented Generation (RAG) in a low lexical diversity setting, specifically for classifying gas pipe damage descriptions based on their patchability.

Our work introduces a new lexical entropy metric to quantify textual diversity and proposes [+]EXPL, a data re-balancing technique that improves retrieval effectiveness in highly imbalanced datasets. The retrieval system is based on SBERT-NLI embeddings, while Mistral7B serves as the generative model for classification.

## Features
Implements Retrieval-Augmented Generation (RAG) for classification.
Introduces Corpus Lexical Entropy as a diversity metric.
Uses FAISS for fast retrieval of similar cases.
Employs SBERT-NLI embeddings for document representation.
Integrates Mistral7B for classification.
Proposes [+]EXPL, a downsampling strategy to mitigate class imbalance.

## Methodology
### Mathematical Formulation
The retrieval and classification processes follow these key steps:

Encoding: Documents and queries are embedded into a vector space using SBERT-NLI.
Retrieval: FAISS retrieves the top-k most similar cases using the dot product as a similarity metric.
Generation: Mistral7B processes the retrieved examples and classifies the query as either patchable (YES) or non-patchable (NO).
Mathematically, the retrieval function selects k nearest documents:

___

where 
e is the encoder function and 
s is the similarity function.

Corpus Lexical Entropy
We define Corpus Lexical Entropy as a measure of term diversity across the corpus:

____

where H(t) is the Shannon entropy of term t, capturing its distribution across documents.

### Retriever
The retrieval system is built upon:

Embedding function: SBERT-NLI
Similarity metric: Internal dot product
Memory selection: FAISS for fast indexing
Decoder
The decoder (θ) is selected based on its ability to classify gas pipe damage. Mistral7B is chosen for its strong domain knowledge and few-shot learning capabilities.

## Experiments
Data
The dataset consists of 11,904 cases describing gas pipe damage conditions, with only 1.06% being patchable. We apply stratified sampling to create training and test sets.

Evaluation Metrics
We use:

F1-Macro score to balance class imbalance effects.
Self-Consistency score (SC) to measure model stability.

Results
SBERT-NLI outperformed MPNet in retrieval, demonstrating better recognition of subtle logical cues (e.g., negations, pressure levels).
Mistral7B achieved 0.68 F1-Macro in zero-shot classification, improving to 0.87 with retrieval augmentation (RAG-[+]EXPL, k=9).
[+]EXPL significantly improved retrieval effectiveness, removing overly similar negative cases near positive ones.

## Conclusion
This work demonstrates that Retrieval-Augmented Generation (RAG) is effective in low-lexical-diversity settings. Key findings include:

SBERT-NLI is the best retrieval encoder for this dataset.
Mistral7B performs well with few-shot learning, especially when [+]EXPL improves retrieval quality.
Fine-tuning was intentionally avoided to maintain generalizability to human-written texts.
Future directions include:

Combining SBERT and MPNet for hybrid retrieval.
Evaluating the model on real-world, multilingual datasets.
Incorporating explanations in the output.

## References
Banghao Chen et al., "Unleashing the potential of prompt engineering in large language models," arXiv:2310.14735 (2023).
Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," (2021).
Nils Reimers and Iryna Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP (2019).
Claude Shannon, "A Mathematical Theory of Communication," The Bell System Technical Journal (1948).




# Retrieval-Augmented Generation in a Low Lexical Diversity Scenario
## Abstract

This research focuses on the application of Retrieval-Augmented Generation (RAG) in a low-lexical-diversity setting, with a focus on classifying gas pipe damage descriptions for patch applicability. To quantify lexical similarities between documents, we introduce a new metric called lexical entropy, while to address class imbalance, we propose [+]EXPL, a data re-balancing technique designed to enhance the retrieval of patchable [+] cases. K-Nearest-Neighbors guide our selection of Sentence-BERT-NLI (SBERT-NLI) as the preferred encoder, as it proves more effective than state-of-the-art models in Semantic Search by capturing crucial nuanced differences between documents. Our results show that Mistral-7B exhibits some understanding of gas leak patchability with an F1-Macro score of 0.68, which improves to 0.87 when supported by the retrieval system. Given the synthetic nature of the documents, we avoid fine-tuning due to concerns that such an approach may not generalize well to human-written descriptions, where vocabulary variety would likely be higher. Nonetheless, RAG-driven approaches show promise in semantically constrained environments.

## Introduction
The objective of this study is to explore the application of Retrieval-Augmented Generation (RAG) models in scenarios characterized by low lexical variety. Specifically, we aim to develop a chatbot to assist gas fitters in determining the applicability of Patch Madflex® for gas pipe repairs, leveraging historical case data. This task focuses on classifying textual descriptions of gas pipe damage as repairable or not, employing foundational models without fine-tuning to evaluate their capability to support real-world tasks using pre-trained knowledge.

The dataset provided by Corporate Research (CoRe) consists of synthetic descriptions derived from tabular data, resulting in a corpus with a high degree of similarity among documents, where vocabulary often mirrors the names of original tabular features. To characterize this phenomenon, we introduce a metric for lexical variety, which quantifies the entropy of the corpus' vocabulary.

Foundation models that are specialized in Semantic Textual Similarity (STS) and are often used in RAG systems may not perform as effectively when all the documents fall within the same semantic sphere, as is the case in our corpus. In our dataset, the documents tend to be highly semantically similar, which makes traditional STS models less effective. Instead, our corpus appears to align better with a Natural Language Inference (NLI) task, where the model must retrieve documents that either confirm or contradict a given query. This alignment with NLI tasks suggests that using models trained for NLI could be more appropriate for this domain.

Among the foundation models trained for NLI, the way the model encodes the final hidden layer plays a crucial role. Specifically, the choice of pooling strategy (e.g., mean pooling) directly affects how tokens associated with specific features in the tabular data are incorporated into the document embeddings. We hypothesize that a more refined pooling strategy—such as treating structured features as distinct contributions to the document vector—may improve performance in low-lexical-variety settings. In this work, we benchmark the superiority of MEAN over CLS pooling for collections where documents exhibit minimal variability in lexical content, as an initial step toward developing a learnable pooling function.

To evaluate the effectiveness of embedding models for information retrieval in scenarios like ours, we assess their ability to retrieve documents from a set of k examples that are semantically similar to the query document $d_q$. To this end, we employ a K-Nearest Neighbors (KNN) retrieval system, which helps identify the most effective combinations of encoder models and similarity metrics. The KNN approach enables us to capture domain-specific nuances in the dataset and refine our retrieval methodology.
Among the encoder models tested, BERT base demonstrates the best performance. 

Our corpus presents two main challenges: (1) a significant class imbalance, with only 1\% of damage descriptions labeled as repairable, and (2) low lexical diversity, which complicates both the classification and retrieval tasks. To address these challenges, we propose a novel method called positive explosion. This method adjusts the dataset by treating positive instances (i.e., repairable damage descriptions) as cluster centers and selectively down-sampling nearby negative examples. This strategy increases the likelihood that positive examples are presented to the large language models (LLMs), improving the model's ability to learn from these more informative instances.

Finally, damage classification is performed using Retrieval-Augmented Generation (RAG) , a framework that integrates retrieval and generative components. This approach enables the model to not only make predictions but also generate explanations for patch applicability. We anticipate that this method will outperform the baseline established during the embedding model selection phase, enhancing the pipeline's overall utility in low-lexical-variety settings.

## Lexical Variety.
Our goal is to define a metric that capures how much the lexicon variates across documents d in a corpus D.

A document can be seen as a collection of terms:

$$d = \{t_1,t_2, ... t_n\}$$

The vocabulary of a corpus is then made by all different terms that appear in D:

$$V_D = \{t_1,...,t_n\}$$

Let $p(t)$ be the probability of find $t$ in a document d:

$$p(t) := \frac{|\{d \in D: t \in d\}|}{|D|}$$

The information content of t according to Shannon is:

$$I(t) = \log_2\left(\frac{1}{p(t)}\right)$$

Under the (although strong in linguistic) assumption that finding $x \in d$ does not influence the presence of $y \in d$:

$$I(\{x,y\}) = \log_2\left(\frac{1}{p(x,y)}\right) = \log_2\left(\frac{1}{p(x)}\frac{1}{p(y)}\right) = \log\left(\frac{1}{p(x)}\right) + \log\left(\frac{1}{p(y)}\right)$$

We define **lexicon variety** as the expected vocabulary's information content in a corpus D:

$$L(V_D) := E[I(V_D)] =  \sum_{t \in V}p(t) \log_{2}\left(\frac{1}{p(t)}\right)$$

- if a term appears in _all_ documents, it's contribute is 0;
- if a term does not appear in the collection, it's contribute is also 0;
- the maximum value is proportional to the length of the vocabulary;
- the more a term belongs to a smaller subset of D, the more it contributes to I(V)

## NLI over STS
-> to be filled. 

## Pooling Matters.
In corpora where the vocabulary remains constant across documents, we hypothesize that a document $\vec{d}$ of length $t$ can be expressed as:  
$$
\vec{d} = \alpha \vec{\text{CLS}} + \sum_{i=1}^{F}\beta_i \vec{e}_i + \sum_{j=1}^{t-F-1}\gamma_j \vec{r}_j,
$$   where:  
- $\vec{\text{CLS}}$: encodes the core semantic content (e.g., gas pipe repairs).  
- $\vec{e}_i$: Embeddings for $F$ structured features derived from the corpus, such as $\vec{e}_{\text{pressure}}$, $\vec{e}_{\text{corrosion}}$, or $\vec{e}_{\text{damage}}$.  
- $\vec{r}_j$: Residual embeddings for tokens outside the structured features. The residual component is labeled as such because we assume that the tokens in $\vec{r}$ shaped the $\vec{e}$ they were associated with, thanks to the attention mechanism.

In our documents, the primary meaningful contributions to $\vec{v}$ come from $\vec{\text{CLS}}$ and $\vec{e}_i$, while the residual component $\sum_{j=1}^{t-F-1}\gamma_j \vec{r}_j$ adds less significant information.
By expliciting cls and mean pooling sentence embedding:
1. **CLS Pooling**: $\vec{d}_{\text{CLS}} = \vec{\text{CLS}}$
2. **Mean Pooling**: $
\vec{d}_{\text{Mean}} = \frac{1}{t + 1} \vec{\text{CLS}} + \frac{1}{t + 1}\sum_{i=1}^{t} \vec{token}_i $
Mean pooling seems to align more to our ideal model. Thus, we argue that it may be worth to seek for a learnable pooling matrix in such scenario, if mean pooling appears to be clearly superior to CLS 

## Data
Data regards damaged gas pipe repaired through Patch Madflex®, a new material developed by Composite Research (CoRe) and tested by ItalGas.

The dataset has been synthetic created from tabular data. Those data, represents the knowledge of a company about a technique of gas pipe repairs: patching. Data was collected as a mix of boolean (e.g.high pressure, bad conditions_) and categorical (e.g.  _damage type, exposure) features. From those feature, an LLM was asked to generate a synthetic textual description `Summary`, along with a rephrased version of it `Summary_1`, which we left unused because of the high similarity with its original version that makes any evaluation on it unreliable. Whether the reparation by Patch Madflex® was successful or not is collected in  `Successful`. 

After dropping some costant features regarding time and locations of the fault, along with links to meta-data not provided with the dataset such as images of the damage, our dataset counts _11904_ examples having _15_ feature each. 

### Exploratory Data Analysis (EDA)
Labels are unfairly distributed (99% _not patchable_, 1% _patchable_). We argue that this originates from the need of CoRe to better the technology, keeping track of faults more than successes. 
The lack of successful reparations and the low lexical entropy of our corpus may cause (-)es shadowing over (+)es. Therefore, downsampling company memory could produce more accurate results while evaluating our system, mitigating this problem.

Some heuristics that precisely clusters `Successful` applications can be found by splitting positive (patchable, +) and negative (not patchable, -) labels.

|index|Feature|Unique Values|Patchable \(+\)|Not Patchable \(-\)|Values Excluded in \(+\)|
|---|---|---|---|---|---|
|5|Severe\_Corrosion|2|1|2|True|
|6|Pipe\_Covered|2|1|2|True|
|7|Branch\_Near\_Fault|2|1|2|True|
|8|High\_Pressure|2|1|2|True|
|9|Damaged\_Valve|2|1|2|True|
|10|Ribs|2|1|2|True|
|3|Damage\_Type|11|9|11|Sheared linear lesion,Visible deformation in axial direction|
|0|Kit\_Size\_num|5|5|5|None|
|1|Pipe\_In\_Bad\_Conditions|2|2|2|None|
|2|Pipe\_Material|6|6|6|None|
|4|Pipe\_Exposure|3|3|3|None|

Patchable faults do not show _damaged valve_, _high pressure_ or _severe corrosion_. Furtherly, there is nor the presence of a _wall_ s nearby patchable damages, nor _ribs_ or _branches_. Moreover, exploring `damage_type`, one can notice that whenever the fault is about _sheared linear lesions_ or _deformations in axial directions_, the attempt to fix the damage by patch was _not successful_. Our goal is to test the capability of a chatbot to address the patchability of a gas fault. For this reason, we **did not** provide heuristics in the prompt. Indeed, the idea is to create a systems that updates with company experience: heuristics may change as the technology improves. 

We augmented the cardinality of the dataset by further generating:
- `[model_name]-Explanation`: description written by Llama3.2 and Mistral when provided with pairs (`Summary`,`Successful`) and asked to explain the outcome (i.e. _repairable_ or not). This will be used to compare the description generated when the model knows about the label vs when the model is asked to infer the label;
- `italian_Summary`: translation of `Summary` in italian, created by means of [Helsinki-NLP/opus-mt-tc-big-en-it](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-it), an open-source Neural Machine Translation model (NMT) for translating from English (en) to Italian (it). This will be the source for multi-lingual testings.

### RAG Database (DB) and Query Set (Q) split
Due to (-)es bias, we split DB/Q in a stratified fashion. Therefore, we create from DATA two sub-groups, namely (-)es and (+)es, then we take the 80% of each group ( 80% of not patchable items, 80% of patchable items) for DB and leave the 20% Q.

Due to CoLab time-limitations, we are forced to downsample the query test: while selecting the totality of patchable damages in the query set (i.e. #25 cases), we downsample to #100 the number of non-patchable items. Therefore, **all tests in this work had run on #125 examples (#25 (+), #100 (-)), excluded from DB**. We leave an exhaustive evaluation of Q for future works. 

## Corpus
Descriptions of gas pipe faults make up our corpus (features `Summary` and `italian_Summary`).

_patchable_ (+) `:

    A galvanized nipple fault has occurred in an aerial pipe due to incorrect installation techniques. The joint was improperly tightened during assembly, which led to excessive flexing and eventually caused the connection point between the two pipes to fail. This resulted in the failure of the entire system and caused significant damage to surrounding components.

_non-patchable_ (-) :

    Sheared linear lesion at user connections, polyethylene material, no strong corrosion, and no high-pressure in the pipe. There is a branch near the break, but the pipe is not covered by a wall and does not have any valves nearby. No ribs are present.

Is worth to notice that summaries **does not mention** whether the damage had been successfully repaired or not.
