[**中文说明**](./README.md) | [**English**](./README_EN.md)

## Chinese ELECTRA
Google and Stanford University released a new pre-trained model called ELECTRA, which has a much compact model size and relatively competitive performance compared to BERT and its variants.
For further accelerating the research of the Chinese pre-trained model, the Joint Laboratory of HIT and iFLYTEK Research (HFL) has released the Chinese ELECTRA models based on the official code of ELECTRA.
ELECTRA-small could reach similar or even higher scores on several NLP tasks with only 1/10 parameters compared to BERT and its variants.

This project is based on the official code of ELECTRA: [https://github.com/google-research/electra](https://github.com/google-research/electra)

----

[Chinese LERT](https://github.com/ymcui/LERT) | [Chinese/English PERT](https://github.com/ymcui/PERT) [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [TextBrewer](https://github.com/airaria/TextBrewer) | [TextPruner](https://github.com/airaria/TextPruner)

More resources by HFL: https://github.com/ymcui/HFL-Anthology

## News
<b>2022/10/29 We release a new pre-trained model called LERT, check https://github.com/ymcui/LERT/</b>

2022/3/30 We release a new pre-trained model called PERT, check https://github.com/ymcui/PERT 

2021/12/17 We release a model pruning toolkit - TextPruner, check https://github.com/airaria/TextPruner

Dec 13, 2020 We release Chinese legal ELECTRA series, check [Download](#Download), [Results for Crime Prediction](#Results-for-Crime-Prediction).

Oct 22, 2020 ELECTRA-180g released, which were trained with high-quality CommonCrawl data, check [Download](#Download).

Sep 15, 2020 Our paper ["Revisiting Pre-Trained Models for Chinese Natural Language Processing"](https://arxiv.org/abs/2004.13922) is accepted to [Findings of EMNLP](https://2020.emnlp.org) as a long paper.

<details>
<summary>Past News</summary>
August 27, 2020 We are happy to announce that our model is on top of GLUE benchmark, check [leaderboard](https://gluebenchmark.com/leaderboard).

May 29, 2020 We have released Chinese ELECTRA-large/small-ex models, check [Download](#Download). We are sorry that only Google Drive links are available at present.

April 7, 2020 PyTorch models are available through [🤗Transformers](https://github.com/huggingface/transformers), check [Quick Load](#Quick-Load)

March 31, 2020  The models in this repository now can be easily accessed through [PaddleHub](https://github.com/PaddlePaddle/PaddleHub), check [Quick Load](#Quick-Load)

March 25, 2020  We have released Chinese ELECTRA-small/base models, check [Download](#Download).
</details>

## Guide
| Section | Description |
|-|-|
| [Introduction](#Introduction) | Introduction to the ELECTRA |
| [Download](#Download) | Download links for Chinese ELECTRA models |
| [Quick Load](#Quick-Load) | Learn how to quickly load our models through [🤗Transformers](https://github.com/huggingface/transformers) or [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) |
| [Baselines](#Baselines) | Baseline results on MRC, Text Classification, etc. |
| [Usage](#Usage) | Detailed instructions on how to use ELECTRA |
| [FAQ](#FAQ) | Frequently Asked Questions |
| [Citation](#Citation) | Citation |


## Introduction
**ELECTRA** provides a new pre-training framework, including two components: **Generator** and **Discriminator**.
- **Generator**: a small MLM that predicts [MASK] to its original token. The generator will replace some of the tokens in the input text.
- **Discriminator**: detect whether the input token is replaced. ELECTRA uses a pre-training task called Replaced Token Detection (RTD) instead of the Masked Language Model (MLM), which is used by BERT and its variants. Note that there is no Next Sentence Prediction (NSP) applied in ELECTRA.

After the pre-training stage, we only use Discriminator for finetuning the downstream tasks.

For more technical details, please check the paper: [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)

![](./pics/model.png)


## Download
We provide TensorFlow models at the moment.

* **`ELECTRA-large, Chinese`**: 24-layer, 1024-hidden, 16-heads, 324M parameters   
* **`ELECTRA-base, Chinese`**：12-layer, 768-hidden, 12-heads, 102M parameters   
* **`ELECTRA-small-ex, Chinese`**: 24-layer, 256-hidden, 4-heads, 25M parameters
* **`ELECTRA-small, Chinese`**: 12-layer, 256-hidden, 4-heads, 12M parameters


#### New Version (180G data)

| Model | Google Drive | Baidu Disk | Size |
| :------- | :---------: | :---------: | :---------: |
| **`ELECTRA-180g-large, Chinese`** | [TensorFlow](https://drive.google.com/file/d/1P9yAuW0-HR7WvZ2r2weTnx3slo6f5u9q/view?usp=sharing) | [TensorFlow（pw:2v5r）](https://pan.baidu.com/s/13UJIG2G0lASjjCvPmh13RQ?pwd=2v5r) | 1G |
| **`ELECTRA-180g-base, Chinese`** | [TensorFlow](https://drive.google.com/file/d/1RlmfBgyEwKVBFagafYvJgyCGuj7cTHfh/view?usp=sharing) | [TensorFlow（pw:3vg1）](https://pan.baidu.com/s/15PQdeh7nRxCgXp9YmjqgsQ?pwd=3vg1) | 383M |
| **`ELECTRA-180g-small-ex, Chinese`** | [TensorFlow](https://drive.google.com/file/d/1NYJTKH1dWzrIBi86VSUK-Ml9Dsso_kuf/view?usp=sharing) | [TensorFlow（pw:93n8）](https://pan.baidu.com/s/1UV83d2LNp5HHwK7X14HjPQ?pwd=93n8) | 92M |
| **`ELECTRA-180g-small, Chinese`** | [TensorFlow](https://drive.google.com/file/d/177EVNTQpH2BRW-35-0LNLjV86MuDnEmu/view?usp=sharing) | [TensorFlow（pw:k9iu）](https://pan.baidu.com/s/1J5DXcehcNtX0iBXNRKLWBw?pwd=k9iu) | 46M |

#### Basic Version (20G data)

| Model | Google Drive | Baidu Disk | Size |
| :------- | :---------: | :---------: | :---------: |
| **`ELECTRA-large, Chinese`** | [TensorFlow](https://drive.google.com/file/d/1ny0NMLkEWG6rseDLiF_NujdHxDcIN51m/view?usp=sharing) | [TensorFlow（pw:1e14）](https://pan.baidu.com/s/1M5pSqDRbb3Vsv5r3TfviBQ?pwd=1e14) | 1G |
| **`ELECTRA-base, Chinese`** | [TensorFlow](https://drive.google.com/open?id=1FMwrs2weFST-iAuZH3umMa6YZVeIP8wD) | [TensorFlow（pw:f32j）](https://pan.baidu.com/s/1HOzCBNaoIEULj_s-q3dDzA?pwd=f32j) | 383M |
| **`ELECTRA-small-ex, Chinese`** | [TensorFlow](https://drive.google.com/file/d/1LluPORc7xtFmCTFR4IF17q77ip82i7__/view?usp=sharing) | [TensorFlow（pw:gfb1）](https://pan.baidu.com/s/1dOLw4feMJcsgZL07V-koWA?pwd=gfb1) | 92M |
| **`ELECTRA-small, Chinese`** | [TensorFlow](https://drive.google.com/open?id=1uab-9T1kR9HgD2NB0Kz1JB_TdSKgJIds) | [TensorFlow（pw:1r4r）](https://pan.baidu.com/s/1UIosBYOHVA3bDuJrFqU0NQ?pwd=1r4r) | 46M |

#### Legal Version (new)

| Model | Google Drive | Baidu Disk | Size |
| :------- | :---------: | :---------: | :---------: |
| **`legal-ELECTRA-large, Chinese`** | [TensorFlow](https://drive.google.com/file/d/1jPyVi_t4QmTkFy7PD-m-hG-lQ8cIETzD/view?usp=sharing) | [TensorFlow（pw:q4gv）](https://pan.baidu.com/s/180cloQ0A3m3VqpLPeKpPYg?pwd=q4gv) | 1G |
| **`legal-ELECTRA-base, Chinese`** | [TensorFlow](https://drive.google.com/file/d/12ZLaoFgpqGJxSi_9KiQV-jdVN4XRGMiD/view?usp=sharing) | [TensorFlow（pw:8gcv）](https://pan.baidu.com/s/1OWwSsr-jCWq3vb7Js4B2vg?pwd=8gcv) | 383M |
| **`legal-ELECTRA-small, Chinese`** | [TensorFlow](https://drive.google.com/file/d/1arQ5qNTNoc1OyMH8wBUKdTMy2QponIFY/view?usp=sharing) | [TensorFlow（pw:kmrj）](https://pan.baidu.com/s/1FIblX4EU23KSQWft3DWL0g?pwd=kmrj) | 46M |


### PyTorch Version
If you need these models in PyTorch,

1) Convert TensorFlow checkpoint into PyTorch, using [🤗Transformers](https://github.com/huggingface/transformers). Please use the script [convert_electra_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_electra_original_tf_checkpoint_to_pytorch.py) provided by 🤗Transformers. For example,
```bash
python transformers/src/transformers/convert_electra_original_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path ./path-to-large-model/ \
--config_file ./path-to-large-model/discriminator.json \
--pytorch_dump_path ./path-to-output/model.bin \
--discriminator_or_generator discriminator
```

2) Download from https://huggingface.co/hfl

Steps: select one of the model in the page above → click "list all files in model" at the end of the model page → download bin/json files from the pop-up window

### Note
The users from Mainland China are encouraged to use iFLYTEK Cloud download links, and the others may use Google Drive links.
The ZIP package includes the following files (For example, `ELECTRA-small, Chinese`):
```
chinese_electra_small_L-12_H-256_A-4.zip
    |- checkpoint                           # checkpoint
    |- electra_small.data-00000-of-00001    # Model weights
    |- electra_small.meta                   # Meta info
    |- electra_small.index                  # Index info
    |- vocab.txt                            # Vocabulary
```


### Training Details
We use the same data for training [RoBERTa-wwm-ext model series](https://github.com/ymcui/Chinese-BERT-wwm), which includes 5.4B tokens.
We also use the same vocabulary from Chinese BERT, which has 21128 tokens.
Other details and hyperparameter settings are listed below (others are remain default):
- `ELECTRA-large`: 24-layers, 1024-hidden, 16-heads, lr: 2e-4, batch: 96, max_len: 512, 2M steps
- `ELECTRA-base`: 12-layers, 768-hidden, 12-heads, lr: 2e-4, batch: 256, max_len: 512, 1M steps
- `ELECTRA-small-ex`: 24-layers, 256-hidden, 4-heads, lr: 5e-4, batch: 384, max_len: 512, 2M steps
- `ELECTRA-small`: 12-layers, 256-hidden, 4-heads, lr: 5e-4, batch: 1024, max_len: 512, 1M steps


## Quick Load
### Huggingface-Transformers

With [Huggingface-Transformers 2.8.0](https://github.com/huggingface/transformers), the models in this repository could be easily accessed and loaded through the following codes.
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME) 
```

The actual model and its `MODEL_NAME` are listed below.

| Original Model | Component | MODEL_NAME |
| - | - | - |
| ELECTRA-180g-large, Chinese | discriminator | hfl/chinese-electra-180g-large-discriminator |
| ELECTRA-180g-large, Chinese | generator | hfl/chinese-electra-180g-large-generator |
| ELECTRA-180g-base, Chinese | discriminator | hfl/chinese-electra-180g-base-discriminator |
| ELECTRA-180g-base, Chinese | generator | hfl/chinese-electra-180g-base-generator |
| ELECTRA-180g-small-ex, Chinese | discriminator | hfl/chinese-electra-180g-small-ex-discriminator |
| ELECTRA-180g-small-ex, Chinese | generator | hfl/chinese-electra-180g-small-ex-generator |
| ELECTRA-180g-small, Chinese | discriminator | hfl/chinese-electra-180g-small-discriminator |
| ELECTRA-180g-small, Chinese | generator | hfl/chinese-electra-180g-small-generator |
| ELECTRA-large, Chinese | discriminator | hfl/chinese-electra-large-discriminator |
| ELECTRA-large, Chinese | generator | hfl/chinese-electra-large-generator |
| ELECTRA-base, Chinese | discriminator | hfl/chinese-electra-base-discriminator |
| ELECTRA-base, Chinese | generator | hfl/chinese-electra-base-generator |
| ELECTRA-small-ex, Chinese | discriminator | hfl/chinese-electra-small-ex-discriminator |
| ELECTRA-small-ex, Chinese | generator | hfl/chinese-electra-small-ex-generator |
| ELECTRA-small, Chinese | discriminator | hfl/chinese-electra-small-discriminator |
| ELECTRA-small, Chinese | generator | hfl/chinese-electra-small-generator |

Legal Version：

| Original Model | Component | MODEL_NAME |
| - | - | - |
| legal-ELECTRA-large, Chinese | discriminator | hfl/chinese-legal-electra-large-discriminator |
| legal-ELECTRA-large, Chinese | generator | hfl/chinese-legal-electra-large-generator |
| legal-ELECTRA-base, Chinese | discriminator | hfl/chinese-legal-electra-base-discriminator |
| legal-ELECTRA-base, Chinese | generator | hfl/chinese-legal-electra-base-generator |å
| legal-ELECTRA-small, Chinese | discriminator | hfl/chinese-legal-electra-small-discriminator |
| legal-ELECTRA-small, Chinese | generator | hfl/chinese-legal-electra-small-generator |



### PaddleHub
With [PaddleHub](https://github.com/PaddlePaddle/PaddleHub), we can download and install the model with one line of code.

```
import paddlehub as hub
module = hub.Module(name=MODULE_NAME)
```

The actual model and its `MODULE_NAME` are listed below.

| Original Model| MODULE_NAME |
| - | - |
| ELECTRA-base | [chinese-electra-base](https://paddlepaddle.org.cn/hubdetail?name=chinese-electra-base&en_category=SemanticModel) |
| ELECTRA-small  | [chinese-electra-small](https://paddlepaddle.org.cn/hubdetail?name=chinese-electra-small&en_category=SemanticModel) |


## Baselines
We compare our Chinese ELECTRA models with [`BERT-base`](https://github.com/google-research/bert)、[`BERT-wwm`, `BERT-wwm-ext`, `RoBERTa-wwm-ext`, `RBT3`](https://github.com/ymcui/Chinese-BERT-wwm) on six tasks.
- [**CMRC 2018 (Cui et al., 2019)**：Span-Extraction Machine Reading Comprehension (Simplified Chinese)](https://github.com/ymcui/cmrc2018)
- [**DRCD (Shao et al., 2018)**：Span-Extraction Machine Reading Comprehension (Traditional Chinese)](https://github.com/DRCSolutionService/DRCD)
- [**XNLI (Conneau et al., 2018)**：Natural Langauge Inference](https://github.com/google-research/bert/blob/master/multilingual.md)
- [**ChnSentiCorp**：Sentiment Analysis](https://github.com/pengming617/bert_classification)
- [**LCQMC (Liu et al., 2018)**：Sentence Pair Matching](http://icrc.hitsz.edu.cn/info/1037/1146.htm)
- [**BQ Corpus (Chen et al., 2018)**：Sentence Pair Matching](http://icrc.hitsz.edu.cn/Article/show/175.html)

For ELECTRA-small/base model, we use the learning rate of `3e-4` and `1e-4` according to the original paper.
**Note that we did NOT tune the hyperparameters w.r.t each task, so it is very likely that you will have better scores than ours.**
To ensure the stability of the results, we run 10 times for each experiment and report the maximum and average scores (in brackets).

### CMRC 2018
[CMRC 2018 dataset](https://github.com/ymcui/cmrc2018) is released by the Joint Laboratory of HIT and iFLYTEK Research. The model should answer the questions based on the given passage, which is identical to [SQuAD](http://arxiv.org/abs/1606.05250). Evaluation metrics: EM / F1

| Model | Development | Test | Challenge | #Params |
| :------- | :---------: | :---------: | :---------: | :---------: |
| BERT-base | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) | 102M |
| BERT-wwm | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) | 102M |
| BERT-wwm-ext | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) | 102M |
| RoBERTa-wwm-ext | 67.4 (66.5) / 87.2 (86.5) | 72.6 (71.4) / 89.4 (88.8) | 26.2 (24.6) / 51.0 (49.1) | 102M |
| RBT3 | 57.0 / 79.0 | 62.2 / 81.8 | 14.7 / 36.2 | 38M |
| **ELECTRA-small** | 63.4 (62.9) / 80.8 (80.2) | 67.8 (67.4) / 83.4 (83.0) | 16.3 (15.4) / 37.2 (35.8) | 12M |
| **ELECTRA-180g-small** | 63.8 / 82.7 | 68.5 / 85.2 | 15.1 / 35.8 | 12M |
| **ELECTRA-small-ex** | 66.4 / 82.2 | 71.3 / 85.3 | 18.1 / 38.3 | 25M |
| **ELECTRA-180g-small-ex** | 68.1 / 85.1 | 71.8 / 87.2 | 20.6 / 41.7 | 25M |
| **ELECTRA-base** | 68.4 (68.0) / 84.8 (84.6) | 73.1 (72.7) / 87.1 (86.9) | 22.6 (21.7) / 45.0 (43.8) | 102M |
| **ELECTRA-180g-base** | 69.3 / 87.0 | 73.1 / 88.6 | 24.0 / 48.6 | 102M |
| **ELECTRA-large** | 69.1 / 85.2 | 73.9 / 87.1 | 23.0 / 44.2 | 324M |
| **ELECTRA-180g-large** | 68.5 / 86.2 | 73.5 / 88.5 | 21.8 / 42.9 | 324M |

### DRCD
[DRCD](https://github.com/DRCKnowledgeTeam/DRCD) is also a span-extraction machine reading comprehension dataset, released by Delta Research Center. The text is written in Traditional Chinese. Evaluation metrics: EM / F1

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) | 102M |
| BERT-wwm | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) | 102M |
| BERT-wwm-ext | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) | 102M |
| RoBERTa-wwm-ext | 86.6 (85.9) / 92.5 (92.2) | 85.6 (85.2) / 92.0 (91.7) | 102M |
| RBT3 | 76.3 / 84.9 | 75.0 / 83.9 | 38M |
| **ELECTRA-small** | 79.8 (79.4) / 86.7 (86.4) | 79.0 (78.5) / 85.8 (85.6) | 12M |
| **ELECTRA-180g-small** | 83.5 / 89.2 | 82.9 / 88.7 | 12M |
| **ELECTRA-small-ex** | 84.0 / 89.5 | 83.3 / 89.1 | 25M |
| **ELECTRA-180g-small-ex** | 87.3 / 92.3 | 86.5 / 91.3 | 25M |
| **ELECTRA-base** | 87.5 (87.0) / 92.5 (92.3) | 86.9 (86.6) / 91.8 (91.7) | 102M |
| **ELECTRA-180g-base** | 89.6 / 94.2 | 88.9 / 93.7 | 102M |
| **ELECTRA-large** | 88.8 / 93.3 | 88.8 / 93.6 | 324M |
| **ELECTRA-180g-large** | 90.1 / 94.8 | 90.5 / 94.7 | 324M |

### XNLI
We use [XNLI](https://github.com/google-research/bert/blob/master/multilingual.md) data for testing the NLI task. Evaluation metric: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 77.8 (77.4) | 77.8 (77.5) | 102M |
| BERT-wwm | 79.0 (78.4) | 78.2 (78.0) | 102M |
| BERT-wwm-ext | 79.4 (78.6) | 78.7 (78.3) | 102M |
| RoBERTa-wwm-ext | 80.0 (79.2) | 78.8 (78.3) | 102M |
| RBT3 | 72.2 | 72.3 | 38M |
| **ELECTRA-small** | 73.3 (72.5) | 73.1 (72.6) | 12M |
| **ELECTRA-180g-small** | 74.6 | 74.6 | 12M |
| **ELECTRA-small-ex** | 75.4 | 75.8 | 25M |
| **ELECTRA-180g-small-ex** | 76.5 | 76.6 | 25M |
| **ELECTRA-base** | 77.9 (77.0) | 78.4 (77.8) | 102M |
| **ELECTRA-180g-base** | 79.6 | 79.5 | 102M |
| **ELECTRA-large** | 81.5 | 81.0 | 324M |
| **ELECTRA-180g-large** | 81.2 | 80.4 | 324M |


### ChnSentiCorp
We use [ChnSentiCorp](https://github.com/pengming617/bert_classification) data for testing sentiment analysis. Evaluation metric: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 94.7 (94.3) | 95.0 (94.7) | 102M |
| BERT-wwm | 95.1 (94.5) | 95.4 (95.0) | 102M |
| BERT-wwm-ext | 95.4 (94.6) | 95.3 (94.7) | 102M |
| RoBERTa-wwm-ext | 95.0 (94.6) | 95.6 (94.8) | 102M |
| RBT3 | 92.8 | 92.8 | 38M |
| **ELECTRA-small** | 92.8 (92.5) | 94.3 (93.5) | 12M |
| **ELECTRA-180g-small** | 94.1 | 93.6 | 12M |
| **ELECTRA-small-ex** | 92.6 | 93.6 | 25M |
| **ELECTRA-180g-small-ex** | 92.8 | 93.4 | 25M |
| **ELECTRA-base** | 93.8 (93.0) | 94.5 (93.5) | 102M |
| **ELECTRA-180g-base** | 94.3 | 94.8 | 102M |
| **ELECTRA-large** | 95.2 | 95.3 | 324M |
| **ELECTRA-180g-large** | 94.8 | 95.2 | 324M |


### LCQMC
[**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.htm) is a sentence pair matching dataset, which could be seen as a binary classification task. Evaluation metric: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT | 89.4 (88.4) | 86.9 (86.4) | 102M |
| BERT-wwm | 89.4 (89.2) | 87.0 (86.8) | 102M |
| BERT-wwm-ext | 89.6 (89.2) | 87.1 (86.6) | 102M |
| RoBERTa-wwm-ext | 89.0 (88.7) | 86.4 (86.1) | 102M |
| RBT3 | 85.3 | 85.1 | 38M |
| **ELECTRA-small** | 86.7 (86.3) | 85.9 (85.6) | 12M |
| **ELECTRA-180g-small** | 86.6 | 85.8 | 12M |
| **ELECTRA-small-ex** | 87.5 | 86.0 | 25M |
| **ELECTRA-180g-small-ex** | 87.6 | 86.3 | 25M |
| **ELECTRA-base** | 90.2 (89.8) | 87.6 (87.3) | 102M |
| **ELECTRA-180g-base** | 90.2 | 87.1 | 102M |
| **ELECTRA-large** | 90.7 | 87.3 | 324M |
| **ELECTRA-180g-large** | 90.3 | 87.3 | 324M |

### BQ Corpus 
[**BQ Corpus**](http://icrc.hitsz.edu.cn/Article/show/175.html) is a sentence pair matching dataset, which could be seen as a binary classification task. Evaluation metric: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT | 86.0 (85.5) | 84.8 (84.6) | 102M |
| BERT-wwm | 86.1 (85.6) | 85.2 (84.9) | 102M |
| BERT-wwm-ext | 86.4 (85.5) | 85.3 (84.8) | 102M |
| RoBERTa-wwm-ext | 86.0 (85.4) | 85.0 (84.6) | 102M |
| RBT3 | 84.1 | 83.3 | 38M |
| **ELECTRA-small** | 83.5 (83.0) | 82.0 (81.7) | 12M |
| **ELECTRA-180g-small** | 83.3 | 82.1 | 12M |
| **ELECTRA-small-ex** | 84.0 | 82.6 | 25M |
| **ELECTRA-180g-small-ex** | 84.6 | 83.4 | 25M |
| **ELECTRA-base** | 84.8 (84.7) | 84.5 (84.0) | 102M |
| **ELECTRA-180g-base** | 85.8 | 84.5 | 102M |
| **ELECTRA-large** | 86.7 | 85.1 | 324M |
| **ELECTRA-180g-large** | 86.4 | 85.4 | 324M |


### Results for Crime Prediction
We adopt CAIL 2018 [crime prediction](https://github.com/liuhuanyong/CrimeKgAssitant) to evaluate the performance of legal ELECTRA. Initial learning rates for small/base/large are 5e-4/3e-4/1e-4.
Evaluation metric: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| ELECTRA-small | 78.84 | 76.35 | 12M |
| **legal-ELECTRA-small** | **79.60** | **77.03** | 12M |
| ELECTRA-base | 80.94 | 78.41 | 102M |
| **legal-ELECTRA-base** | **81.71** | **79.17** | 102M |
| ELECTRA-large | 81.53 | 78.97 | 324M |
| **legal-ELECTRA-large** | **82.60** | **79.89** | 324M |


## Usage
Users may utilize the ELECTRA for fine-tuning their own tasks.
Here we only illustrate the basic usage, and the users are encouraged to refer to the [official guidelines](https://github.com/google-research/electra) as well.

In this tutorial, we will use `ELECTRA-small` model for finetuning CMRC 2018 task. 
- `data-dir`：working directory
- `model-name`：model name, here we set as `electra-small`
- `task-name`：task name, here we set as `cmrc2018`. Our codes are adapted for all six tasks, where the `task-name`s are `cmrc2018`，`drcd`，`xnli`，`chnsenticorp`，`lcqmc`，`bqcorpus`.

### Step 1: Download and unzip model
Download ELECTRA-small model from [Download](#Download) section, and unzip the files into `${data-dir}/models/${model-name}`.
The folder should contain five files, including `electra_model.*`, `vocab.txt`, `checkpoint`.

### Step 2: Prepare for task data
Download [CMRC 2018 training and development data](https://github.com/ymcui/cmrc2018/tree/master/squad-style-data), and rename them as `train.json`, `dev.json`.
Put two files into `${data-dir}/finetuning_data/${task-name}` directory.

### Step 3: Run command
```shell
python run_finetuning.py \
    --data-dir ${data-dir} \
    --model-name ${model-name} \
    --hparams params_cmrc2018.json
```
The `data-dir` and `model-name` are illustrated in previous steps. `hparams` is a JSON dictionary. In this tutorial, `params_cmrc2018.json` includes the hyperparameter settings for finetuning.
```json
{
    "task_names": ["cmrc2018"],
    "max_seq_length": 512,
    "vocab_size": 21128,
    "model_size": "small",
    "do_train": true,
    "do_eval": true,
    "write_test_outputs": true,
    "num_train_epochs": 2,
    "learning_rate": 3e-4,
    "train_batch_size": 32,
    "eval_batch_size": 32,
}
```
In this JSON file, we only listed some of the important hyperparameters.
For all hyperparameter entries, please check [configure_finetuning.py](./configure_finetuning.py)。

After running the program,
1. For machine reading comprehension tasks, the predicted JSON file `cmrc2018_dev_preds.json` will be saved in `${data-dir}/results/${task-name}_qa/`. You can use evaluation script to get the final scores, such as `python cmrc2018_drcd_evaluate.py dev.json cmrc2018_dev_preds.json`
2. For text classification tasks, the accuracy will be printed on the screen right away, such as `xnli: accuracy: 72.5 - loss: 0.67`

## FAQ
**Q: How to set learning rate in finetuning stage?**  
A: We recommend to use the learning rate in the paper as default (3e-4 for small, 1e-4 for base), and adjust according to your own task.
Note that the initial learning rate may be higher than that in BERT or RoBERTa.

**Q: Do you have PyTorch models?**  
A: Yes. You can check [Download](#Download).

**Q: Is it possible to share the training data?**  
A: I am sorry that it is not possible.

**Q: Do you have any future plans?**  
A: Stay tuned!

## Citation
If you find the technical report or resource is useful, please cite our work in your paper.
- Primary: https://ieeexplore.ieee.org/document/9599397  
```
@journal{cui-etal-2021-pretrain,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2021},
  url={https://ieeexplore.ieee.org/document/9599397},
  doi={10.1109/TASLP.2021.3124365},
 }
```
- Secondary: https://www.aclweb.org/anthology/2020.findings-emnlp.58
```
@inproceedings{cui-etal-2020-revisiting,
    title = "Revisiting Pre-Trained Models for {C}hinese Natural Language Processing",
    author = "Cui, Yiming  and
      Che, Wanxiang  and
      Liu, Ting  and
      Qin, Bing  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.58",
    pages = "657--668",
}
```

## Follow us
Follow our official WeChat account to keep updated with our latest technologies!

![qrcode.png](./pics/qrcode.jpg)


## Issues
Before you submit an issue:

- **You are advised to read [FAQ](https://github.com/ymcui/MacBERT#FAQ) first before you submit an issue.**
- Repetitive and irrelevant issues will be ignored and closed by [stable-bot](stale · GitHub Marketplace). Thank you for your understanding and support.
- We cannot acommodate EVERY request, and thus please bare in mind that there is no guarantee that your request will be met.
- Always be polite when you submit an issue.
