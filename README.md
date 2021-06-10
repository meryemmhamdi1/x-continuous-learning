# Cross-lingual Continuous Learning
This is to analyze and visualize different parameters and components in the architectures of different cross-lingual downstream tasks to detect/mitigate catastrophic forgetting patterns and improve generalization.

## Different Setups:
### Setup 1: CIL: 
Cross-CIL, Fixed LL: "Monolingual CIL" high to low 
Stream consisting of different combinations of classes from single language each time.
We then average over all languages.
Each stream consists of one language each time 

### Setup 2: CLL: 
Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning"
Stream consisting of different combinations of languages.
Each stream sees all intents
### Setup 3: CIL-CLL : 
Cross-CIL-LL: "Cross-lingual combinations of languages/intents"
Stream consisting of different combinations
- Matrix of batches of languages and classes:
    - Horizontally goes linearly over all intents in the batch of each languages batch before moving to the next languages batch
    - Vertically goes linearly over all languages of each intent batch before moving to the next intents batch

### Setup 4: Multi-Task:
Multi-tasking one joint model over all languages and classes (upper bound)
Working on reproducing the results for MTOP (multi-task) setup.

