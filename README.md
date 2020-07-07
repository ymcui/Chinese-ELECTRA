## ELECTRA-Hongkongese and XLNet-Hongkongese

Google and Stanford University released a transformer model called [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) which is modeled as a 
discriminator. This allows for fewer parameters and can be trained with lower compute to get to the same results as BERT.

This repo is for the ELECTRA models pretrained from purely Hong Kong data only. In addition to a Traditional Chinese question answering dataset, three Hongkongese evaluation dataset were created to explore what they can do. Training was done with the [official ELECTRA repo](https://github.com/google-research/electra). This repo was forked from [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) to reuse the evaluation code. 

[XLNet](https://arxiv.org/abs/1906.08237) is an auto-regressive model trained to use bidirectional context. It was shown to surpass BERT in many tasks. In addition, it is capable of generating text sequences. The base version of XLNet was trained using the same data as ELECTRA with the [official repo](https://github.com/zihangdai/xlnet).

All the models are available through [Transformers](https://github.com/huggingface/transformers/) under [toastynews](https://huggingface.co/toastynews). Please consult their [documentation](https://huggingface.co/transformers/) for detail on usage. For ELECTRA models, normally only the discriminator is used.
 * [xlnet-hongkongese-base](https://huggingface.co/toastynews/xlnet-hongkongese-base)
 * [electra-hongkongese-small-discriminator](https://huggingface.co/toastynews/electra-hongkongese-small-discriminator)
 * [electra-hongkongese-small-generator](https://huggingface.co/toastynews/electra-hongkongese-small-generator)
 * [electra-hongkongese-base-discriminator](https://huggingface.co/toastynews/electra-hongkongese-base-discriminator)
 * [electra-hongkongese-base-generator](https://huggingface.co/toastynews/electra-hongkongese-base-generator)
 * [electra-hongkongese-large-discriminator](https://huggingface.co/toastynews/electra-hongkongese-large-discriminator)
 * [electra-hongkongese-large-generator](https://huggingface.co/toastynews/electra-hongkongese-large-generator)

 *Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)*


## Guide
| Section | Description |
|-|-|
| [Details](#Details) | Details of these models |
| [Baselines](#Baselines) | Baseline results on MRC, Text Classification, etc. |


## Details
Basics technical information for the models. All models were trained on a single TPUv3.
| Value            | XLNet Base |ELECTRA Small | ELELCTRA Base | ELELCTRA Large|
| :--------------- |----------: |------------: |-------------: |-------------: |
| Tokens           |       507M |         507M |          507M |          507M |
| Max Sequence Length |     512 |          512 |           512 |           512 |
| Parameters       |       117M |          12M |          102M |          324M |
| Batch Size       |         32 |          384 |           256 |            96 |
| Vocab Size       |      32000 |         30000|          30000|          30000|
| Tokenizer        | SentencePiece |  WordPiece|      WordPiece|      WordPiece|

Models were trained with the following mix of data.
| Data                                              |   % |
| ------------------------------------------------- | --: |
| News Articles / Blogs                             | 58% |
| Yue Wikipedia / EVCHK                             | 18% |
| Restaurant Reviews                                | 12% |
| Forum Threads                                     | 12% |
| Online Fiction                                    |  1% |

The following is the distribution of different languages within the corpus.
| Language                                          |   % |
| ------------------------------------------------- | --: |
| Standard Chinese                                  | 62% |
| Hongkongese                                       | 30% |
| English                                           |  8% |


## Baselines
The models were tested on four Traditional Chinese tasks. Results of Chinese ELECTRA are included for comparison.
- [**DRCD (Shao et al., 2018)**：Span-Extraction Machine Reading Comprehension (Traditional Chinese)](https://github.com/DRCSolutionService/DRCD)
- [**OpenRice Sentiment Analysis**](https://github.com/toastynews/openrice-senti) - new Hongkongese task
- [**LIHKG Categorization**](https://github.com/toastynews/lihkg-cat-v2) - new Hongkongese task
- [**Words HK Semantic Similarity**](https://github.com/toastynews/wordshk-sem) - new Hongkongese task

For ELECTRA-small model, the learning rate was set at `3e-4` according to the original paper, the others were set at default. To ensure the stability of the results, each task was run 10 times. The maximum and average (in brackets) scores are reported below.


### DRCD
[**DRCD**](https://github.com/DRCKnowledgeTeam/DRCD) is a span-extraction machine reading comprehension dataset, released by Delta Research Center. The text is written in Traditional Chinese. DRCD is based on Chinese Wikipedia which is not part of the training data for the Hongkongese models. Evaluation metrics: EM / F1

| Model | Development | Test | Cleanliness |
| :------- | :---------: | :---------: | :-------: |
| XLNet-base-Chinese | 83.8 (83.2) / 92.3 (92.0) | 83.5 (82.8) / 92.2 (91.8) | Dirty |
| **XLNet-base-Hongkongese** | 79.1 (78.2) / 79.1 (78.2) | 76.5 (76.1) / 76.5 (76.1) | Clean |
| ELECTRA-small-Chinese | 79.8 (79.4) / 86.7 (86.4) | 79.0 (78.5) / 85.8 (85.6) | Dirty |
| **ELECTRA-small-Hongkongese** | 78.1 (77.7) / 85.6 (85.4) | 77.1 (76.7) / 84.8 (84.4) | Clean |
| ELECTRA-base-Chinese | 87.5 (87.0) / 92.5 (92.3) | 86.9 (86.6) / 91.8 (91.7) | Dirty |
| **ELECTRA-base-Hongkongese** | 84.0 (83.7) / 90.2 (90.0) | 83.6 (83.0) / 90.1 (89.6) | Clean |
| ELECTRA-large-Chinese | 88.8 / 93.3 | 88.8 / 93.6 | Dirty |
| **ELECTRA-large-Hongkongese** | 86.2 (85.7) / 92.2 (91.8) | 85.4 (84.7) / 91.1 (90.9) | Clean |


### OpenRice Senti
[**OpenRice Senti**](https://github.com/toastynews/openrice-senti) is a new dataset for Hongkongese sentiment analysis. This dataset is part of training data for the Hongkongese models so the results should be considered dirty. Evaluation metrics: Accuracy

| Model | Development | Test | Cleanliness |
| :------- | :---------: | :---------: | :-------: |
| XLNet-base-Chinese | 80.5 (80.0) | 80.5 (79.8) | Clean |
| **XLNet-base-Hongkongese** | 81.7 (81.2) | 82.2 (81.4) | Dirty |
| ELECTRA-small-Chinese | 78.8 (77.8) | 78.8 (77.9) | Clean |
| **ELECTRA-small-Hongkongese** | 80.3 (79.6) | 79.8 (79.0) | Dirty |
| ELECTRA-base-Chinese | 78.9 (78.1) | 79.6 (79.1) | Clean |
| **ELECTRA-base-Hongkongese** | 83.8 (83.1) | 82.0 (81.5) | Dirty |
| ELECTRA-large-Chinese | 82.3 (81.1) |80.9 (79.8) | Clean |
| **ELECTRA-large-Hongkongese** | 83.0 (82.5) | 80.0 (79.7) | Dirty |


### LIHKG Cat
[**LIHKG Cat**](https://github.com/toastynews/lihkg-cat-v2) is a new dataset for Hongkongese multi-class classification task. This dataset is part of training data for the Hongkongese models so the results should be considered dirty. Evaluation metrics: Accuracy

| Model | Development | Test | Cleanliness |
| :------- | :---------: | :---------: | :-------: |
| XLNet-base-Chinese | 71.7 (70.6) | 71.4 (70.7) | Clean |
| **XLNet-base-Hongkongese** | 71.1 (70.7) | 70.2 (69.5) | Dirty |
| ELECTRA-small-Chinese | 64.4 (63.3) | 64.5 (63.7) | Clean |
| **ELECTRA-small-Hongkongese** | 64.1 (63.1) | 63.1 (62.6) | Dirty |
| ELECTRA-base-Chinese | 67.9 (67.1) | 68.3 (67.4) | Clean |
| **ELECTRA-base-Hongkongese** | 70.1 (69.0) | 70.9 (70.0) | Dirty |
| ELECTRA-large-Chinese | 69.1 (68.4) | 71.4 (70.4) | Clean |
| **ELECTRA-large-Hongkongese** | 70.0 (69.1) | 71.8 (69.9) | Dirty |


### WordsHK Sem
[**WordsHK Sem**](https://github.com/toastynews/wordshk-sem) is a new dataset for Hongkongese semantic similarity task. This dataset is not part of training data for the Hongkongese models. Evaluation metrics: Accuracy

| Model | Development | Test | Cleanliness |
| :------- | :---------: | :---------: | :-------: |
| XLNet-base-Chinese | 81.9 (71.8) 76.9* | 83.1 (72.0) 78.9* | Clean |
| **XLNet-base-Hongkongese** | 66.7 (66.7) 88.9* | 66.7 (66.7) 87.3* | Clean |
| ELECTRA-small-Chinese | 80.7 (80.0) | 80.0 (79.2) | Clean |
| **ELECTRA-small-Hongkongese** | 81.7 (81.1) | 82.0 (80.0) | Clean |
| ELECTRA-base-Chinese | 88.0 (87.6) | 88.8 (88.1) | Clean |
| **ELECTRA-base-Hongkongese** | 90.2 (89.3) | 90.7 (90.1) | Clean |
| ELECTRA-large-Chinese | 91.3 (90.6) | 90.9 (90.4) | Clean |
| **ELECTRA-large-Hongkongese** | 92.6 (91.4) | 92.3 (91.5) | Clean |

\* With the default of 3 epoches, 6 of 10 XLNet-base-Chinese finetuned models have accuracy of 66.7 (always negative baseline). All XLNet-base-Hongkongese finetuned models have accuracy of 66.7. The \* values are the accuracy after 24 epoches.
