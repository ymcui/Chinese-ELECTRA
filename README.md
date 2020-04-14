## ELECTRA-Hongkongese

TL;DR Use [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) for most use cases. This model is useful only for Hongkongese model research.

Google and Stanford University released a new transformer model called [ELECTRA]((https://openreview.net/pdf?id=r1xMH1BtvB)) which is modeled as a 
discriminator. This allows for much fewer parameters and can be trained with much lower compute. Official code of ELECTRA: [https://github.com/google-research/electra](https://github.com/google-research/electra)

The Joint Laboratory of HIT and iFLYTEK Research (HFL) has released the [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) models based on the official code of ELECTRA. The Chinese-ELECTRA models were trained with a lot of data and compute.

This repo is about a small model I pretrained from only Hong Kong data. Using the same evaluation tasks as Chinese-ELECTRA, with two additional Hongkongese ones, I want to explore what it can do. Training was done with the official ELECTRA code. This repo was forked from Chinese-ELECTRA to use the code for evaluation. 

As can be seen in the [Baselines](#Baselines) section, Chinese-ELECTRA beats this model in every test. Surprisingly, this model also do quite well with the simplified Chinese tasks. It seems that the ELECTRA model can overcome a lot of the differences between senmantics and character sets. For most use cases, it is probably better to use the Chinese-ELECTRA pretrained models. I recommend only to use this model for Hongkongese research. The model can be found in the [Download](#Download) section if you want to experiment it yourself.


## Guide
| Section | Description |
|-|-|
| [Download](#Download) | Download links for ELECTRA Hongkongese model |
| [Baselines](#Baselines) | Baseline results on MRC, Text Classification, etc. |
| [Usage](#Usage) | Detailed instructions on how to use ELECTRA |


## Download
I provide only the small TensorFlow model at the moment.

* **`ELECTRA-small, Hongkongese`**: 12-layer, 256-hidden, 4-heads, 12M parameters

| Model | Google Drive | Size |
| :------- | :--------- | :---------: |
| **`ELECTRA-small, Hongkongese`** | [TensorFlow](https://drive.google.com/file/d/10PD5tXTXF3dn23ypuqTXne5Vo-2EaMbK/view?usp=sharing) | 159MB |

The ZIP package includes the following files (For example, `ELECTRA-small, Hongkongese`):
```
electra_hongkongese
    |- checkpoint                                # Checkpoint
    |- graph.pbtxt                               # Graph
    |- model.ckpt-1000000.data-00000-of-00001    # Model weights
    |- model.ckpt-1000000.index                  # Index info
    |- model.ckpt-1000000.meta                   # Meta info    
    |- vocab.txt                                 # Vocabulary
```

### Training Details
I used the the following mix of data, total about 362M tokens.
| Data                                                                             |   % |
| -------------------------------------------------------------------------------- | --: |
| [LIHKG](https://lihkg.com/)                                                      |  7% |
| [The Encyclopedia of Virtual Communities in Hong Kong](https://evchk.wikia.org/) |  3% |
| [Toasty News](https://www.toastynews.com/)                                       | 85% |
| [Yue Wikipedia](https://zh-yue.wikipedia.org/)                                   |  5% |

Percentage of Hongkongese (Hong Kong Cantonese) within the corpus is around 10%. To give some perspective, in the [OSCAR](https://traces1.inria.fr/oscar/) extraction of [Common Crawl](https://commoncrawl.org/) data, the Chinese data is 249GB in size, of which only 30MB is Hongkongese, around 0.01%. 

The vocabulary is copied from Chinese BERT, which has 21128 tokens, but I left `vocab_size` at default. 
The only non-default parameter is `train_batch_size` and `eval_batch_size` at 96 so I can fit it in my GPU.
All other parameters were left at default.

Comparing ELECTRA Hongkongese to Chinese ELECTRA:
| Parameter  |    Chinese | Hongkongese |          % | Notes          | 
| :--------- | ---------: | ----------: | ---------: | -------------- |
| Tokens     |       5.4B |        362M |         7% |                |
| batch_size |       1024 |          96 |         9% |                |
| max_seq_length |    512 |         128 |        32% |                |
| vocab_size |      21128 |       30522 |       100% | Same vocab.txt |


## Baselines
I compared ELECTRA Hongkongese with Chinese ELECTRA small model on seven tasks.
- [**CMRC 2018 (Cui et al., 2019)**：Span-Extraction Machine Reading Comprehension (Simplified Chinese)](https://github.com/ymcui/cmrc2018)
- [**DRCD (Shao et al., 2018)**：Span-Extraction Machine Reading Comprehension (Traditional Chinese)](https://github.com/DRCSolutionService/DRCD)
- [**XNLI (Conneau et al., 2018)**：Natural Langauge Inference](https://github.com/google-research/bert/blob/master/multilingual.md) - I have no correctly formatted data for this
- [**ChnSentiCorp**：Sentiment Analysis](https://github.com/pengming617/bert_classification)
- [**LCQMC (Liu et al., 2018)**：Sentence Pair Matching](http://icrc.hitsz.edu.cn/info/1037/1146.htm)
- [**BQ Corpus (Chen et al., 2018)**：Sentence Pair Matching](http://icrc.hitsz.edu.cn/Article/show/175.html)
- [**OpenRice for sentiment analysis**](https://github.com/toastynews/openrice-senti) - new Hongkongese task
- [**LIHKG for categorization**](https://github.com/toastynews/lihkg-cat) - new Hongkongese task

For ELECTRA-small model, I used the learning rate of `3e-4` according to the original paper.
**Note that I did NOT tune the hyperparameters w.r.t each task, so it is very likely that you will have better scores than ours.**
To ensure the stability of the results, I ran 10 times for each experiment and report the maximum and average scores (in brackets).

### CMRC 2018
[CMRC 2018 dataset](https://github.com/ymcui/cmrc2018) is released by the Joint Laboratory of HIT and iFLYTEK Research. The model should answer the questions based on the given passage, which is identical to [SQuAD](http://arxiv.org/abs/1606.05250). Evaluation metrics: EM / F1

| Model | Development | Test | Challenge | #Params |
| :------- | :---------: | :---------: | :---------: | :---------: |
| BERT-base | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) | 102M | 
| BERT-wwm | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) | 102M | 
| BERT-wwm-ext | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) | 102M | 
| RoBERTa-wwm-ext | 67.4 (66.5) / 87.2 (86.5) | 72.6 (71.4) / 89.4 (88.8) | 26.2 (24.6) / 51.0 (49.1) | 102M | 
| RBT3 | 57.0 / 79.0 | 62.2 / 81.8 | 14.7 / 36.2 | 38M |
| ELECTRA-base-Chinese | 68.4 (68.0) / 84.8 (84.6) | 73.1 (72.7) / 87.1 (86.9) | 22.6 (21.7) / 45.0 (43.8) | 102M |
| **ELECTRA-small-Chinese** | 63.4 (62.9) / 80.8 (80.2) | 67.8 (67.4) / 83.4 (83.0) | 16.3 (15.4) / 37.2 (35.8) | 12M |
| **ELECTRA-small-Hongkongese** | 53.2 (52.8) / 71.0 (70.6) | 44.8 (44.2) / 67.1 (66.4) |  N/A | 12M |


### DRCD
[DRCD](https://github.com/DRCKnowledgeTeam/DRCD) is also a span-extraction machine reading comprehension dataset, released by Delta Research Center. The text is written in Traditional Chinese. Evaluation metrics: EM / F1

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) | 102M | 
| BERT-wwm | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) | 102M | 
| BERT-wwm-ext | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) | 102M | 
| RoBERTa-wwm-ext | 86.6 (85.9) / 92.5 (92.2) | 85.6 (85.2) / 92.0 (91.7) | 102M | 
| RBT3 | 76.3 / 84.9 | 75.0 / 83.9 | 38M |
| ELECTRA-base-Chinese | 87.5 (87.0) / 92.5 (92.3) | 86.9 (86.6) / 91.8 (91.7) | 102M |
| **ELECTRA-small-Chinese** | 79.8 (79.4) / 86.7 (86.4) | 79.0 (78.5) / 85.8 (85.6) | 12M |
| **ELECTRA-small-Hongkongese** | 73.3 (72.5) / 81.4 (80.7) | 71.0 (70.4) / 79.5 (79.0) | 12M |


### XNLI
We use [XNLI](https://github.com/google-research/bert/blob/master/multilingual.md) data for testing the NLI task. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 77.8 (77.4) | 77.8 (77.5) | 102M | 
| BERT-wwm | 79.0 (78.4) | 78.2 (78.0) | 102M | 
| BERT-wwm-ext | 79.4 (78.6) | 78.7 (78.3) | 102M |
| RoBERTa-wwm-ext | 80.0 (79.2) | 78.8 (78.3) | 102M |
| RBT3 | 72.2 | 72.3 | 38M | 
| ELECTRA-base-Chinese | 77.9 (77.0) | 78.4 (77.8) | 102M |
| **ELECTRA-small-Chinese** | 73.3 (72.5) | 73.1 (72.6) | 12M |
| **ELECTRA-small-Hongkongese** | N/A | N/A | 12M |


### ChnSentiCorp
We use [ChnSentiCorp](https://github.com/pengming617/bert_classification) data for testing sentiment analysis. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 94.7 (94.3) | 95.0 (94.7) | 102M |
| BERT-wwm | 95.1 (94.5) | 95.4 (95.0) | 102M |
| BERT-wwm-ext | 95.4 (94.6) | 95.3 (94.7) | 102M |
| RoBERTa-wwm-ext | 95.0 (94.6) | 95.6 (94.8) | 102M |
| RBT3 | 92.8 | 92.8 | 38M | 
| ELECTRA-base-Chinese | 93.8 (93.0) | 94.5 (93.5) | 102M |
| **ELECTRA-small-Chinese** | 92.8 (92.5) | 94.3 (93.5) | 12M |
| **ELECTRA-small-Hongkongese** | 89.7 (88.8) | 90.6 (90.0) | 12M |


### LCQMC
[**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.htm) is a sentence pair matching dataset, which could be seen as a binary classification task. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT | 89.4 (88.4) | 86.9 (86.4) | 102M | 
| BERT-wwm | 89.4 (89.2) | 87.0 (86.8) | 102M |
| BERT-wwm-ext | 89.6 (89.2) | 87.1 (86.6) | 102M |
| RoBERTa-wwm-ext | 89.0 (88.7) | 86.4 (86.1) | 102M |
| RBT3 | 85.3 | 85.1 | 38M |
| ELECTRA-base-Chinese | 90.2 (89.8) | 87.6 (87.3) | 102M |
| **ELECTRA-small-Chinese** | 86.7 (86.3) | 85.9 (85.6) | 12M |
| **ELECTRA-small-Hongkongese** | 84.2 (83.9) | 85.0 (84.8) | 12M |


### BQ Corpus 
[**BQ Corpus**](http://icrc.hitsz.edu.cn/Article/show/175.html) is a sentence pair matching dataset, which could be seen as a binary classification task. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT | 86.0 (85.5) | 84.8 (84.6) | 102M | 
| BERT-wwm | 86.1 (85.6) | 85.2 (84.9) | 102M |
| BERT-wwm-ext | 86.4 (85.5) | 85.3 (84.8) | 102M |
| RoBERTa-wwm-ext | 86.0 (85.4) | 85.0 (84.6) | 102M |
| RBT3 | 84.1 | 83.3 | 38M |
| ELECTRA-base-Chinese | 84.8 (84.7) | 84.5 (84.0) | 102M |
| **ELECTRA-small-Chinese** | 83.5 (83.0) | 82.0 (81.7) | 12M |
| **ELECTRA-small-Hongkongese** | 81.7 (81.0) | 80.6 (80.3) | 12M |


### OpenRice Senti
[**OpenRice Senti**](https://github.com/toastynews/openrice-senti) is a new dataset for Hongkongese sentiment analysis. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| **ELECTRA-small-Chinese** | 69.9 (69.1) | 72.0 (70.9) | 12M |
| **ELECTRA-small-Hongkongese** | 67.9 (67.5) | 70.5 (69.7) | 12M |


### LIHKG Cat
[**LIHKG Cat**](https://github.com/toastynews/lihkg-cat) is a new dataset for Hongkongese multi-class classification task. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| **ELECTRA-small-Chinese** | 60.6 (59.4) | 59.8 (58.4) | 12M |
| **ELECTRA-small-Hongkongese** | 59.4 (57.3) | 58.3 (56.6) | 12M |


## Usage
Users may utilize the ELECTRA for fine-tuning their own tasks.
Here we only illustrate the basic usage, and the users are encouraged to refer to the [official guidelines](https://github.com/google-research/electra) as well.

In this tutorial, we will use the model for finetuning OpenRice Senti task. 
- `data-dir`：working directory
- `model-name`：model name, here we set as `electra_hongkongese_20200328`
- `task-name`：task name, here we set as `openrice-senti`. Our codes are adapted for all seven tasks, where the `task-name`s are `cmrc2018`，`drcd`，`xnli`，`chnsenticorp`，`lcqmc`，`bqcorpus`, `openrice-senti`, `lihkg-cat`.

### Step 1: Download and unzip model
Download model from [Download](#Download) section, and unzip the files into `${data-dir}/models/${model-name}`.
The folder should contain five files, including `model.*`, `vocab.txt`, `checkpoint`.

### Step 2: Prepare for task data
Download [CMRC 2018 training and development data](https://github.com/ymcui/cmrc2018/tree/master/squad-style-data), and rename them as `train.json`, `dev.json`, `eval.json`.
Put two files into `${data-dir}/models/${task-name}` directory.

### Step 3: Run command
```shell
python run_finetuning.py \
    --data-dir ${data-dir} \
    --model-name ${model-name} \
    --hparams params_openrice.json
```
The `data-dir` and `model-name` are illustrated in previous steps. `hparams` is a JSON dictionary. In this tutorial, `params_openrice.json` includes the hyperparameter settings for finetuning.
```json
{
    "task_names": ["openrice-senti"],
    "model_size": "small",
    "learning_rate": 3e-4
}
```
In this JSON file, we only listed some of the important hyperparameters.
For all hyperparameter entries, please check [configure_finetuning.py](./configure_finetuning.py)。

After running the program,
1. For machine reading comprehension tasks, the predicted JSON file `cmrc2018_dev_preds.json` will be saved in `${data-dir}/results/${task-name}_qa/`. You can use evaluation script to get the final scores, such as `python cmrc2018_drcd_evaluate.py dev.json cmrc2018_dev_preds.json`
2. For text classification tasks, the accuracy will be printed on the screen right away, such as `xnli: accuracy: 72.5 - loss: 0.67`
