[**中文说明**](./README.md) | [**English**](./README_EN.md)

## Chinese ELECTRA
Google and Stanford University released a new pre-trained model called ELECTRA, which has a much compact model size and relatively competitive performance compared to BERT and its variants.
For further accelerating the research of the Chinese pre-trained model, the Joint Laboratory of HIT and iFLYTEK Research (HFL) has released the Chinese ELECTRA models based on the official code of ELECTRA.
ELECTRA-small could reach similar or even higher scores on several NLP tasks with only 1/10 parameters compared to BERT and its variants.

This project is based on the official code of ELECTRA: [https://github.com/google-research/electra](https://github.com/google-research/electra)


## News
May 29, 2020 We have released Chinese ELECTRA-large/small-ex models, check [Download](#Download). We are sorry that only Google Drive links are available at present.

April 7, 2020 PyTorch models are available through [🤗Transformers](https://github.com/huggingface/transformers), check [Quick Load](#Quick-Load)

March 31, 2020  The models in this repository now can be easily accessed through [PaddleHub](https://github.com/PaddlePaddle/PaddleHub), check [Quick Load](#Quick-Load)

March 25, 2020  We have released Chinese ELECTRA-small/base models, check [Download](#Download).


## Guide
| Section | Description |
|-|-|
| [Introduction](#Introduction) | Introduction to the ELECTRA |
| [Download](#Download) | Download links for Chinese ELECTRA models |
| [Quick Load](#Quick-Load) | Learn how to quickly load our models through [🤗Transformers](https://github.com/huggingface/transformers) or [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) |
| [Baselines](#Baselines) | Baseline results on MRC, Text Classification, etc. |
| [Usage](#Usage) | Detailed instructions on how to use ELECTRA |
| [FAQ](#FAQ) | Frequently Asked Questions |


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

| Model | Google Drive | iFLYTEK Cloud | Size |
| :------- | :--------- | :---------: | :---------: | 
| **`ELECTRA-large, Chinese (new)`** | [TensorFlow+config](https://drive.google.com/file/d/1ny0NMLkEWG6rseDLiF_NujdHxDcIN51m/view?usp=sharing) | N/A | 1G |
| **`ELECTRA-small-ex, Chinese (new)`** | [TensorFlow+config](https://drive.google.com/file/d/1LluPORc7xtFmCTFR4IF17q77ip82i7__/view?usp=sharing) | N/A | 92M |
| **`ELECTRA-base, Chinese`** | [TensorFlow](https://drive.google.com/open?id=1FMwrs2weFST-iAuZH3umMa6YZVeIP8wD) <br/> [PyTorch-D](https://drive.google.com/open?id=1iBanmudRHLm3b4X4kL_FxccurDjL4RYe) <br/> [PyTorch-G](https://drive.google.com/open?id=1x-fcgS9GU8X51H1FFiqkh0RIDMGTTX7c) | [TensorFlow (pw:3VQu)](https://pan.iflytek.com:443/link/43B111080BD4A2D3370423912B45491E) <br/> [PyTorch-D (pw:WQ8r)](http://pan.iflytek.com:80/link/31F0C2FB919C6099DEC72FD72C0AFCFB) <br/> [PyTorch-G (pw:XxnY)](http://pan.iflytek.com:80/link/2DD6237FE1B99ECD81F775FC2C272149)| 383M |
| **`ELECTRA-small, Chinese`** | [TensorFlow](https://drive.google.com/open?id=1uab-9T1kR9HgD2NB0Kz1JB_TdSKgJIds) <br/> [PyTorch-D](https://drive.google.com/open?id=1A1wdw41kOFC3n3AjfFTRZHQdjCL84bsg) <br/> [PyTorch-G](https://drive.google.com/open?id=1FpdHG2UowDTIepiuOiJOChrtwJSMQJ6N) | [TensorFlow (pw:wm2E)](https://pan.iflytek.com:443/link/E5B4E8FE8B22A5FF03184D34CB2F1767) <br/> [PyTorch-D (pw:Cch4)](http://pan.iflytek.com:80/link/5AE514A3721E4E75A0E04B8E99BB4098) <br/> [PyTorch-G (pw:xCH8)](http://pan.iflytek.com:80/link/CB800D74191E948E06B45238AB797933) | 46M |

*PyTorch-D: discriminator, PyTorch-G: generator

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

For Pytorch version weights, please use the script [convert_electra_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_electra_original_tf_checkpoint_to_pytorch.py) provided by 🤗Transformers. For example,
```bash
python transformers/src/transformers/convert_electra_original_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path ./path-to-large-model/ \
--config_file ./path-to-large-model/discriminator.json \
--pytorch_dump_path ./path-to-output/model.bin \
--discriminator_or_generator discriminator
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
| ELECTRA-large, Chinese | discriminator | hfl/chinese-electra-large-discriminator |
| ELECTRA-large, Chinese | generator | hfl/chinese-electra-large-generator |
| ELECTRA-base, Chinese | discriminator | hfl/chinese-electra-base-discriminator |
| ELECTRA-base, Chinese | generator | hfl/chinese-electra-base-generator |
| ELECTRA-small-ex, Chinese | discriminator | hfl/chinese-electra-small-ex-discriminator |
| ELECTRA-small-ex, Chinese | generator | hfl/chinese-electra-small-ex-generator |
| ELECTRA-small, Chinese | discriminator | hfl/chinese-electra-small-discriminator |
| ELECTRA-small, Chinese | generator | hfl/chinese-electra-small-generator |


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
| **ELECTRA-small-ex** | 66.4 / 82.2 | 71.3 / 85.3 | 18.1 / 38.3 | 25M |
| **ELECTRA-base** | 68.4 (68.0) / 84.8 (84.6) | 73.1 (72.7) / 87.1 (86.9) | 22.6 (21.7) / 45.0 (43.8) | 102M |
| **ELECTRA-large** | 69.1 / 85.2 | 73.9 / 87.1 | 23.0 / 44.2 | 324M |

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
| **ELECTRA-small-ex** | 84.0 / 89.5 | 83.3 / 89.1 | 25M |
| **ELECTRA-base** | 87.5 (87.0) / 92.5 (92.3) | 86.9 (86.6) / 91.8 (91.7) | 102M |
| **ELECTRA-large** | 88.8 / 93.3 | 88.8 / 93.6 | 324M |

### XNLI
We use [XNLI](https://github.com/google-research/bert/blob/master/multilingual.md) data for testing the NLI task. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 77.8 (77.4) | 77.8 (77.5) | 102M | 
| BERT-wwm | 79.0 (78.4) | 78.2 (78.0) | 102M | 
| BERT-wwm-ext | 79.4 (78.6) | 78.7 (78.3) | 102M |
| RoBERTa-wwm-ext | 80.0 (79.2) | 78.8 (78.3) | 102M |
| RBT3 | 72.2 | 72.3 | 38M | 
| **ELECTRA-small** | 73.3 (72.5) | 73.1 (72.6) | 12M |
| **ELECTRA-small-ex** | 75.4 | 75.8 | 25M |
| **ELECTRA-base** | 77.9 (77.0) | 78.4 (77.8) | 102M |
| **ELECTRA-large** | 81.5 | 81.0 | 324M |

### ChnSentiCorp
We use [ChnSentiCorp](https://github.com/pengming617/bert_classification) data for testing sentiment analysis. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT-base | 94.7 (94.3) | 95.0 (94.7) | 102M |
| BERT-wwm | 95.1 (94.5) | 95.4 (95.0) | 102M |
| BERT-wwm-ext | 95.4 (94.6) | 95.3 (94.7) | 102M |
| RoBERTa-wwm-ext | 95.0 (94.6) | 95.6 (94.8) | 102M |
| RBT3 | 92.8 | 92.8 | 38M | 
| **ELECTRA-small** | 92.8 (92.5) | 94.3 (93.5) | 12M |
| **ELECTRA-small-ex** | 92.6 | 93.6 | 25M |
| **ELECTRA-base** | 93.8 (93.0) | 94.5 (93.5) | 102M |
| **ELECTRA-large** | 95.2 | 95.3 | 324M |

### LCQMC
[**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.htm) is a sentence pair matching dataset, which could be seen as a binary classification task. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT | 89.4 (88.4) | 86.9 (86.4) | 102M | 
| BERT-wwm | 89.4 (89.2) | 87.0 (86.8) | 102M |
| BERT-wwm-ext | 89.6 (89.2) | 87.1 (86.6) | 102M |
| RoBERTa-wwm-ext | 89.0 (88.7) | 86.4 (86.1) | 102M |
| RBT3 | 85.3 | 85.1 | 38M |
| **ELECTRA-small** | 86.7 (86.3) | 85.9 (85.6) | 12M |
| **ELECTRA-small-ex** | 87.5 | 86.0 | 25M |
| **ELECTRA-base** | 90.2 (89.8) | 87.6 (87.3) | 102M |
| **ELECTRA-large** | 90.7 | 87.3 | 324M |


### BQ Corpus 
[**BQ Corpus**](http://icrc.hitsz.edu.cn/Article/show/175.html) is a sentence pair matching dataset, which could be seen as a binary classification task. Evaluation metrics: Accuracy

| Model | Development | Test | #Params |
| :------- | :---------: | :---------: | :---------: |
| BERT | 86.0 (85.5) | 84.8 (84.6) | 102M | 
| BERT-wwm | 86.1 (85.6) | 85.2 (84.9) | 102M |
| BERT-wwm-ext | 86.4 (85.5) | 85.3 (84.8) | 102M |
| RoBERTa-wwm-ext | 86.0 (85.4) | 85.0 (84.6) | 102M |
| RBT3 | 84.1 | 83.3 | 38M |
| **ELECTRA-small** | 83.5 (83.0) | 82.0 (81.7) | 12M |
| **ELECTRA-small-ex** | 84.0 | 82.6 | 25M |
| **ELECTRA-base** | 84.8 (84.7) | 84.5 (84.0) | 102M |
| **ELECTRA-large** | 86.7 | 85.1 | 324M |


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
Put two files into `${data-dir}/models/${task-name}` directory.

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


## Follow us
Follow our official WeChat account to keep updated with our latest technologies!

![qrcode.png](./pics/qrcode.jpg)


## Issues
If there is any problem, please submit a GitHub Issue.
