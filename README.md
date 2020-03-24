[**中文说明**](./README.md) | [**English**](./README_EN.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="500"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/ymcui/Chinese-ELECTRA/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-ELECTRA.svg?color=blue&style=flat-square">
    </a>
</p>

谷歌与斯坦福大学共同研发的最新预训练模型ELECTRA因其小巧的模型体积以及良好的模型性能受到了广泛关注。
为了进一步促进中文预训练模型技术的研究与发展，哈工大讯飞联合实验室基于官方ELECTRA训练代码以及大规模的中文数据训练出中文ELECTRA预训练模型供大家下载使用。

本项目基于谷歌&斯坦福大学官方的ELECTRA：[https://github.com/google-research/electra](https://github.com/google-research/electra)


## 新闻
2020/3/24 **Thank you for your attention before our initial release :) We are working on the evaluation at the moment and will keep you updated once we release our models. Hopefully, it won't take too long.**


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍ELECTRA基本原理 |
| [模型下载](#模型下载) | 中文ELECTRA预训练模型下载 |
| [基线系统效果](#基线系统效果) | 中文基线系统效果：阅读理解、文本分类等 |
| [使用方法](#使用方法) | 模型的详细使用方法 |
| [FAQ](#FAQ) | 常见问题答疑 |


## 简介
ELECTRA提出了一套新的预训练框架，其中包括两个部分：**Generator**和**Discriminator**。
- **Generator**: 一个小的MLM，在[MASK]的位置预测原来的词。Generator将用来把输入文本做部分词的替换。
- **Discriminator**: 判断输入句子中的每个词是否被替换，即使用Replaced Token Detection (RTD)预训练任务，取代了BERT原始的Masked Language Model (MLM)。需要注意的是这里并没有Next Sentence Prediction (NSP)任务。

在预训练阶段结束之后，我们只使用Discriminator作为下游任务精调的基模型。

更详细的内容请查阅ELECTRA论文：[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)

![](./pics/model.png)


## 模型下载
TBA


## 基线系统效果
TBA


## 使用方法
TBA


## FAQ
TBA


## 关注我们
欢迎关注**哈工大讯飞联合实验室**官方微信公众号，了解最新的技术动态。

![qrcode.png](./pics/qrcode.jpg)


## 问题反馈
如有问题，请在GitHub Issue中提交。
