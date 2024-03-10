<h1 align="center"> Fast Efficient COVID-19 Case Detection </h1>

<p align="center">
Data modeling submodule for Udacity's Machine Learning Engineering Nanodegree program.
<img src="https://drive.google.com/uc?export=view&id=1GLw6KAaXYa80wSE8NlQsZJWgN9ZLBesY" width="500" />
</p>

 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SL2eCnZYnYd8NCm1xKhEm20mCHjNOhkv?usp=sharing) 
 
## Repo Contents
<img src="https://drive.google.com/uc?export=view&id=13VzZwQerS54SF_6zbJr-_jHAQjqI2W5Z" width="50%" /> 

#### Packages Used
```
pytorchcv
sklearn
fastai #(v1)
numpy
torch
torchvision
tracemalloc
pthflops
plotly_express
```
 
## Overview
The [notebook](https://github.com/codeamt/mle-capstone-modeling/blob/master/mle_capstone_data_modeling.ipynb) in this repo provides detailed steps on building and executing this end-to-end Fast.ai Image Classification pipeline. When running the notebook in a Jupyter environment, be sure to upload and extract the output zip file -- and designate the extracted folder as your path-- from the [data preprocessing submodule](https://github.com/codeamt/mle-capstone-data) as well as [train](https://github.com/codeamt/mle-capstone-modeling/blob/master/train_split_v3.csv) and [test](https://github.com/codeamt/mle-capstone-modeling/blob/master/test_split_v3.csv) .csv label files, and the [common.py](https://github.com/codeamt/mle-capstone-modeling/blob/master/common.py) module to patch the [pytorchcv](https://pypi.org/project/pytorchcv/) lib's Swish Activation implementation with a more memory-efficient one:

<img src="https://drive.google.com/uc?export=view&id=1c3JknK7nAjatZJ_cF9CIC_LE-tNenOIo" width="70%" />

## Model Implementation 

| Base Architecture: CovidNet | Blocks |
|:------:|:---------:|
| <img src="https://drive.google.com/uc?export=view&id=1tIBoHC7jQTfXCJ2HTONeEB9rsobMCScz" width="60%" /> | <img src="https://drive.google.com/uc?export=view&id=144jKu3GiLEiQOeyRTtXO-wgVCldx2mxQ" width="40%" />|
 
 [COVID-Net](https://arxiv.org/pdf/2003.09871.pdf) Illustration  (L. Wang and A. Wong., 2020)

#### Fastai Classifier: 
<img src="https://drive.google.com/uc?export=view&id=1X44o4v4jmyocE1ADjVsLPxoCeTpSqlOJ" width="70%" />

Check out my [project proposal](https://github.com/codeamt/FastEfficientCovidNet/blob/master/proposal.pdf) for more info.

## Training
#### Hardware Specs

Model was trained in Google Colab with the following GPU:

<img src="https://drive.google.com/uc?export=view&id=11jK2HS0vImD6RXOIPdsgOYTbI9bFZiJf" width="50%" />

#### Phases
Each phase utilizes the "[One-cycle training strategy](https://docs.fast.ai/callback.schedule.html#learner.fit_one_cycle)" provided with Fastai learner class.  

| Phase       | Epochs    | Tuned Hyperparams |
|:------:|:---------:|:---------:
| <img src="https://drive.google.com/uc?export=view&id=1MV6vyCI5-B2cZwesNqLOcfaIAygWFLop" width="300" /> |  <img src="https://drive.google.com/uc?export=view&id=1YMpKBHzMt6f_p3K9EAswXyUfljXzejAH" width="400" /> | <ul align="left"><li>State: Unfrozen</li><li>Epochs: 6</li><li>Learning Rate: slice(5e-3/6)</li><li>Weight Decay: 2e-3</li></ul> |
| <img src="https://drive.google.com/uc?export=view&id=1CLOCpFQhiAASAzv9c5-irxNYJdqaXtfR" width="300" /> |  <img src="https://drive.google.com/uc?export=view&id=1Stso2L2lxRNcPg-SogfZpUHmhKKxfJhm" width="500" />|<ul align="left"><li>State: Frozen</li><li>Epochs: 3</li><li>Learning Rate: slice(2e-3)</li><li>Weight Decay: None </li></ul></p>|

| Input Size*       | FLOPs    | GFLOPs |
|:------:|:---------:|:--------:|
| (1, 3, 240, 240) |  718,097,040 | 0.72 | 

* Fore details about FLOPs per layer, see [notebook](https://github.com/codeamt/mle-capstone-modeling/blob/master/project_results/mle_capstone_data_modeling.ipynb) for Fastai Learner Callback and helpers, measuring RAM usage.


#### Performance Metrics
<img src="https://drive.google.com/uc?export=view&id=1Wmf1LFpb4b1N5EvBphQUsi8siur3ipFt" width="30%" />
  

|  Label | Precision | Specificity | Sensitivity | F1 |
|:-----:|:------:|:---------:|:--------:|:-----:|
| Pneumonia |  0.943256  |    0.965595   |    0.949438    | 0.946337 |
| COVID-19 |   0.938776  |     0.998882   |    0.958333    |   0.948454 |
| Normal |   0.966916  |     0.956911   |    0.962112   |   0.964508 |


<img src="https://drive.google.com/uc?export=view&id=1YQUNS0bGA6BGTUdStchSzZqGduZEy7d3" width="30%" />

|  Label | Precision | Specificity | Sensitivity | F1 |
|:-----:|:------:|:---------:|:--------:|:-----:|
| Pneumonia |  0.93178  |    0.958628   |    0.942761    | 0.937238 |
| COVID-19 |   0.965517  |     0.999326   |  0.903226    |   0.933333 |
| Normal |   0.959091  |     0.956911   |    0.953672   |   0.956374 |

## References
[1](https://arxiv.org/pdf/2003.09871.pdf) COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases
from Chest X-Ray Image. L. Wang and A. Wong., 2020.

[2](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi474G8teiEAxUKFVkFHZ1HBDgQFnoECA0QAQ&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1803.09820&usg=AOvVaw20Klrm3A1js9TO0l8qDd6C&opi=89978449) A Disciplined Approach to Neural Network Hyper-Parameters: Part 1 - Learniing Rate, Batch Size, Momentum, and Weight Decay. L Smith., 2018.

#### Resources
- [Fastai Documentation](https://docs.fast.ai/)
