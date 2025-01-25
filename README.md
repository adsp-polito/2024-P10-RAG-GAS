# Retrieval-Augmented Generation on a Domain-Specific Corpus about Gas Pipes Repairs.
_Abstract_ (to be completed)

Retrieval-Augmented Generation (RAG) has demonstrated its effectiveness in tasks requiring precise information retrieval combined with text generation. This paper investigates its application in low lexicon variety settings, where document vocabularies are highly similar across the corpus. Using a dataset of synthetic damage descriptions for gas pipe repairs, we explore how RAG can address the classification of patchable versus non-patchable damages. Our methodology formalizes lexicon diversity and introduces a novel document representation model tailored to low-entropy corpora.

## Introduction
The objective of this study is to explore the application of Retrieval-Augmented Generation (RAG) models in scenarios characterized by low lexical variety. Specifically, we aim to develop a chatbot to assist gas fitters in determining the applicability of Patch Madflex® for gas pipe repairs, leveraging historical case data. This task focuses on classifying textual descriptions of gas pipe damage as repairable or not, employing foundational models without fine-tuning to evaluate their capability to support real-world tasks using pre-trained knowledge.

The dataset provided by Corporate Research (CoRe) consists of synthetic descriptions derived from tabular data, resulting in a corpus with a high degree of similarity among documents, where vocabulary often mirrors the names of original tabular features. To characterize this phenomenon, we introduce a metric for lexical variety, which quantifies the entropy of the corpus' vocabulary.

Foundation models that are specialized in Semantic Textual Similarity (STS) and are often used in RAG systems may not perform as effectively when all the documents fall within the same semantic sphere, as is the case in our corpus. In our dataset, the documents tend to be highly semantically similar, which makes traditional STS models less effective. Instead, our corpus appears to align better with a Natural Language Inference (NLI) task, where the model must retrieve documents that either confirm or contradict a given query. This alignment with NLI tasks suggests that using models trained for NLI could be more appropriate for this domain.

Among the foundation models trained for NLI, the way the model encodes the final hidden layer plays a crucial role. Specifically, the choice of pooling strategy (e.g., mean pooling) directly affects how tokens associated with specific features in the tabular data are incorporated into the document embeddings. We hypothesize that a more refined pooling strategy—such as treating structured features as distinct contributions to the document vector—may improve performance in low-lexical-variety settings. In this work, we benchmark the superiority of MEAN over CLS pooling for collections where documents exhibit minimal variability in lexical content, as an initial step toward developing a learnable pooling function.

To evaluate the effectiveness of embedding models for information retrieval in scenarios like ours, we assess their ability to retrieve documents from a set of k examples that are semantically similar to the query document $d_q$. To this end, we employ a K-Nearest Neighbors (KNN) retrieval system, which helps identify the most effective combinations of encoder models and similarity metrics. The KNN approach enables us to capture domain-specific nuances in the dataset and refine our retrieval methodology.
Among the encoder models tested, BERT base demonstrates the best performance. 

Our corpus presents two main challenges: (1) a significant class imbalance, with only 1\% of damage descriptions labeled as repairable, and (2) low lexical diversity, which complicates both the classification and retrieval tasks. To address these challenges, we propose a novel method called positive explosion. This method adjusts the dataset by treating positive instances (i.e., repairable damage descriptions) as cluster centers and selectively down-sampling nearby negative examples. This strategy increases the likelihood that positive examples are presented to the large language models (LLMs), improving the model's ability to learn from these more informative instances.

Finally, damage classification is performed using Retrieval-Augmented Generation (RAG) , a framework that integrates retrieval and generative components. This approach enables the model to not only make predictions but also generate explanations for patch applicability. We anticipate that this method will outperform the baseline established during the embedding model selection phase, enhancing the pipeline's overall utility in low-lexical-variety settings.


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

We claim that tabular origin and  task-specificity make the corpus' vocabulary small. To prove our hypothesis, we compare `Summary` with a general-purpose corpus such as Common Crawl.  We sample _n = |`Summary`|_ documents from [Common Crawl - News](https://huggingface.co/datasets/vblagoje/cc_news) (CC-News) to prove our hypothesis.  We are interested in quantifying the level of _ lexical diversity_ within corpora. To do so, we define **lexical entropy** of a corpus D having vocabulary $V_D$ as follows:

$$H(D) = -\sum_{t \in V_D}p_tlog(p_t) =  \sum_{t \in V_D} idf(t,D)e^{-idf(t,D)}$$
$$p_t = \frac{|\{d \in D, d :  t \in d\}|}{|D|} = e^{-idf(t,D)}$$

Results in the table below confirms our intuition: the diversity in CC-News is 10 times larger than our corpus. 

|metrics|common\_crawl\_downsampled|gas|
|---|---|---|
|&#124;V&#124;|101993|2556|
|&#124;D&#124;|11904|11904|
|avg\_d\_length|395|70|
|avg\_idf|8\.92|7\.95|
|H\(D\)|346|42\.4|
