# OmiEmbed
***Please also have a look at our brand new omics-to-omics DL freamwork 👀:***
[OmiTrans](https://github.com/zhangxiaoyu11/OmiTrans)

[![DOI](https://zenodo.org/badge/334077812.svg)](https://zenodo.org/badge/latestdoi/334077812)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ce304bf91b534e26b310b3c50072e8ae)](https://www.codacy.com/gh/zhangxiaoyu11/OmiEmbed/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=zhangxiaoyu11/OmiEmbed&amp;utm_campaign=Badge_Grade)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/zhangxiaoyu11/OmiEmbed/blob/main/LICENSE)
![Safe](https://img.shields.io/badge/Stay-Safe-red?logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0iTGF5ZXJfMSIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNTEwIDUxMCIgaGVpZ2h0PSI1MTIiIHZpZXdCb3g9IjAgMCA1MTAgNTEwIiB3aWR0aD0iNTEyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxnPjxnPjxwYXRoIGQ9Im0xNzQuNjEgMzAwYy0yMC41OCAwLTQwLjU2IDYuOTUtNTYuNjkgMTkuNzJsLTExMC4wOSA4NS43OTd2MTA0LjQ4M2g1My41MjlsNzYuNDcxLTY1aDEyNi44MnYtMTQ1eiIgZmlsbD0iI2ZmZGRjZSIvPjwvZz48cGF0aCBkPSJtNTAyLjE3IDI4NC43MmMwIDguOTUtMy42IDE3Ljg5LTEwLjc4IDI0LjQ2bC0xNDguNTYgMTM1LjgyaC03OC4xOHYtODVoNjguMThsMTE0LjM0LTEwMC4yMWMxMi44Mi0xMS4yMyAzMi4wNi0xMC45MiA0NC41LjczIDcgNi41NSAxMC41IDE1LjM4IDEwLjUgMjQuMnoiIGZpbGw9IiNmZmNjYmQiLz48cGF0aCBkPSJtMzMyLjgzIDM0OS42M3YxMC4zN2gtNjguMTh2LTYwaDE4LjU1YzI3LjQxIDAgNDkuNjMgMjIuMjIgNDkuNjMgNDkuNjN6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTM5OS44IDc3LjN2OC4wMWMwIDIwLjY1LTguMDQgNDAuMDctMjIuNjQgNTQuNjdsLTExMi41MSAxMTIuNTF2LTIyNi42NmwzLjE4LTMuMTljMTQuNi0xNC42IDM0LjAyLTIyLjY0IDU0LjY3LTIyLjY0IDQyLjYyIDAgNzcuMyAzNC42OCA3Ny4zIDc3LjN6IiBmaWxsPSIjZDAwMDUwIi8+PHBhdGggZD0ibTI2NC42NSAyNS44M3YyMjYuNjZsLTExMi41MS0xMTIuNTFjLTE0LjYtMTQuNi0yMi42NC0zNC4wMi0yMi42NC01NC42N3YtOC4wMWMwLTQyLjYyIDM0LjY4LTc3LjMgNzcuMy03Ny4zIDIwLjY1IDAgNDAuMDYgOC4wNCA1NC42NiAyMi42NHoiIGZpbGw9IiNmZjRhNGEiLz48cGF0aCBkPSJtMjEyLjgzIDM2MC4xMnYzMGg1MS44MnYtMzB6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTI2NC42NSAzNjAuMTJ2MzBoMzYuMTRsMzIuMDQtMzB6IiBmaWxsPSIjZmZiZGE5Ii8+PC9nPjwvc3ZnPg==)
[![GitHub Repo stars](https://img.shields.io/github/stars/zhangxiaoyu11/OmiEmbed?style=social)](https://github.com/zhangxiaoyu11/OmiEmbed/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/zhangxiaoyu11/OmiEmbed?style=social)](https://github.com/zhangxiaoyu11/OmiEmbed/network/members)

**OmiEmbed: A Unified Multi-task Deep Learning Framework for Multi-omics Data**

**Xiaoyu Zhang** (x.zhang18@imperial.ac.uk)

Data Science Institute, Imperial College London

## Introduction

OmiEmbed is a unified framework for deep learning-based omics data analysis, which supports:

1.  Multi-omics integration
2.  Dimensionality reduction
3.  Omics embedding learning
4.  Tumour type classification
5.  Phenotypic feature reconstruction
6.  Survival prediction
7.  Multi-task learning for aforementioned tasks

Paper Link: [https://doi.org/10.3390/cancers13123047](https://doi.org/10.3390/cancers13123047)

## Getting Started

### Prerequisites
-   CPU or NVIDIA GPU + CUDA CuDNN
-   [Python](https://www.python.org/downloads) 3.6+
-   Python Package Manager
    -   [Anaconda](https://docs.anaconda.com/anaconda/install) 3 (recommended)
    -   or [pip](https://pip.pypa.io/en/stable/installing/) 21.0+
-   Python Packages
    -   [PyTorch](https://pytorch.org/get-started/locally) 1.2+
    -   TensorBoard 1.10+
    -   Tables 3.6+
    -   scikit-survival 0.6+
    -   prefetch-generator 1.0+
-   [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 2.7+

### Installation
-   Clone the repo
```bash
git clone https://github.com/zhangxiaoyu11/OmiEmbed.git
cd OmiEmbed
```
-   Install the dependencies
    -   For conda users  
    ```bash
    conda env create -f environment.yml
    conda activate omiembed
    ```
    -   For pip users
    ```bash
    pip install -r requirements.txt
    ```

### Try it out
-   Train and test using the built-in sample dataset with the default settings
```bash
python train_test.py
```
-   Check the output files
```bash
cd checkpoints/test/
```
-   Visualise the metrics and losses
```bash
tensorboard --logdir=tb_log --bind_all
```

## Citation
If you use this code in your research, please cite our paper.
```bibtex
@Article{OmiEmbed2021,
    AUTHOR = {Zhang, Xiaoyu and Xing, Yuting and Sun, Kai and Guo, Yike},
    TITLE = {OmiEmbed: A Unified Multi-Task Deep Learning Framework for Multi-Omics Data},
    JOURNAL = {Cancers},
    VOLUME = {13},
    YEAR = {2021},
    NUMBER = {12},
    ARTICLE-NUMBER = {3047},
    ISSN = {2072-6694},
    DOI = {10.3390/cancers13123047}
}
```

## OmiTrans
***Please also have a look at our brand new omics-to-omics DL freamwork 👀:***
[OmiTrans](https://github.com/zhangxiaoyu11/OmiTrans)

## License
This source code is licensed under the [MIT](https://github.com/zhangxiaoyu11/OmiEmbed/blob/main/LICENSE) license.
