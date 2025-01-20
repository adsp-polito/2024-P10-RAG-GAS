# Retrieval-Augmented Generation on a Synthetic Domain-Specific Corpus about Gas Pipes Repairs Through Patching.
``

In this work, we explore Retrieval-Augmented Generation (*RAG*)  on a synthetic-domain specific corpus.

## Data
Data regards damaged gas pipe repaired through Patch Madflex®, a new material developed by Composite Research (CoRe) and tested by ItalGas.

The dataset has been synthetic created from tabular data. Those data, represents the knowledge of a company about a technique of gas pipe repairs: patching. Data was collected as a mix of boolean (e.g.high pressure, bad conditions_) and categorical (e.g.  _damage type, exposure) features. From those feature, an LLM was asked to generate a synthetic textual description `Summary`, along with a rephrased version of it `Summary_1`, which we left unused because of the high similarity with its original version that makes any evaluation on it unreliable. Whether the reparation by Patch Madflex® was successful or not is collected in  `Successful`. 

After dropping some costant features regarding time and locations of the fault, along with links to meta-data not provided with the dataset such as images of the damage, our dataset counts _11904_ examples having _15_ feature each. 

### Exploratory Data Analysis (EDA)
Labels are unfairly distributed (99% _not patchable_, 1% _patchable_). We argue that this originates from the need of CoRe to better the technology, keeping track of faults more than successes. 
The lack of successful reparations and the low lexical entropy of our corpus may cause (-)es shadowing over (+)es. Therefore, downsampling company memory could produce more accurate results while evaluating our system, mitigating this problem.

Some heuristics that precisely clusters `Successful` applications can be found by splitting positive (patchable, +) and negative (not patchable, -) labels.

|index|Feature|COUNT|Patchable \(+\)|Not Patchable \(-\)|Values Excluded in \(+\)|
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

Patchable faults do not show _damaged valve_, _high pressure_ or _severe corrosion_. Furtherly, there is nor the presence of a _wall_ s nearby patchable damages, nor _ribs_ or _branches_. Moreover, exploring `damage_type`, one can notice that whenever the fault is about _sheared linear lesions_ or _deformations in axial directions_, the attempt to fix the damage by patch was _not successful_. Our goal is to test the capability of a chatbot to address the patchability of a gas fault. For this reaso, we **did not** provide heuristics in the prompt. Indeed, the idea is to create a systems that updates with company experience: heuristics may change as the technology improves. 

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

We claim that tabular origin and  task-specificity make the corpus' vocabulary small. To prove our hypothesis, we compare `Summary` with a general-purpose corpus such as Common Crawl.  We sample _n = |`Summary`|_ documents from Common Crawl - News (CC-News) to prove our hypothesis.  We are interested in quantify the level of _ lexical diversity_ within corpora. To do so, we define **lexical entropy** of a corpus D having vocabulary $V_D$ as follows:
$$p_t = \frac{|\{d \in D, d :  t \in d\}|}{|D|} = e^{-idf(t,D)}$$
$$H(D) = -\sum_{t \in V_D}p_tlog(p_t) =  \sum_{t \in V_D} idf(t,C)e^{-idf(t,D)}$$ 

Results in the table below confirms our intuition: the diversity in CC-News is 10 times larger than our corpus. 

|metrics|common\_crawl\_downsampled|gas|
|---|---|---|
|&#124;V&#124;|101993|2556|
|&#124;D&#124;|11904|11904|
|avg\_d\_length|395|70|
|avg\_idf|8\.92|7\.95|
|H\(D\)|346|42\.4|

## Retrieval System
