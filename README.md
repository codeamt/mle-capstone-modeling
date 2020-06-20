# Fast Efficient COVID-19 Case Detection
Data modeling submodule for Udacity's Machine Learning Engineering Nanodegree program.
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SL2eCnZYnYd8NCm1xKhEm20mCHjNOhkv?usp=sharing) 
 
## Repo Contents
<img src="https://drive.google.com/uc?export=view&id=13VzZwQerS54SF_6zbJr-_jHAQjqI2W5Z" width="50%" /> 
 
## Instructions
The [notebook](https://github.com/codeamt/mle-capstone-modeling/blob/master/mle_capstone_data_modeling.ipynb) in this repo provides detailed steps on building and executing this Image Classification pipeline. When running the notebook in a Jupyter environment, be sure to upload and extract the output zip file -- and designate the extracted folder as your path-- from the [data preprocessing submodule](https://github.com/codeamt/mle-capstone-data) as well as [train](https://github.com/codeamt/mle-capstone-modeling/blob/master/train_split_v3.csv) and [test](https://github.com/codeamt/mle-capstone-modeling/blob/master/test_split_v3.csv) .csv label files, and the [common.py](https://github.com/codeamt/mle-capstone-modeling/blob/master/common.py) module to patch the [pytorchcv](https://pypi.org/project/pytorchcv/) lib's Swish Activation implementation with a more memory-efficient one:

<img src="https://drive.google.com/uc?export=view&id=122cw8-wh1bH1eoTWtqAG7NKoCOADcjv-" width="60%" />

## Model Implementation 
### Base Architecture
 <img src="https://drive.google.com/uc?export=view&id=1dt7jsm-hWLb83XRUc2LZGE6lg8KDCq0X" width="70%" />
 
 [COVID-Net](https://arxiv.org/pdf/2003.09871.pdf) Illustration  (L. Wang and A. Wong., 2020)
 
#### Blocks
<img src="https://drive.google.com/uc?export=view&id=1MfMDu7eVSgYJu7X5iVR34WNKCOI3WYzo" width="60%" />
 
#### Layers 
<img src="https://drive.google.com/uc?export=view&id=1V_Y4ORfyQR_YZA3D5ZIe5Z2IqYQqhjlh" width="50%" />
 
### Classifier
#### Illustration
<img src="https://drive.google.com/uc?export=view&id=1C5hGu7-x9w_ry8DVSwdTE5BT47YNzJIc" width="80%" />

#### Summary of Classifier Layers:
<img src="https://drive.google.com/uc?export=view&id=1j29FdE5yG91qm6IdbeIccAWB9xsrxic1" width="80%" />

#### Benchmark Comparison
<img src="https://drive.google.com/uc?export=view&id=1kSrGJ6qTIAdRX5sYY7kUpPtpeZvnrscM" width="80%" /> 

## Training
#### Hardware Specs
<img src="https://drive.google.com/uc?export=view&id=1HezHQPn9Dnx_-qOI9wepe00fyRCrXIcG" width="70%" />

#### Phases
<img src="https://drive.google.com/uc?export=view&id=1PKaYBZtfExdsvOzAhXa9u0CmLNyfQMMK" width="70%" />
 
#### Confusion Matrix
<img src="https://drive.google.com/uc?export=view&id=1mG-jtWGuEtle06DH8LgZkyboUmsVkdrR" width="50%" />

#### Training Performance Metrics
<img src="https://drive.google.com/uc?export=view&id=1S6aePVF8kEeXDg0lsDYsKHGLQDRuKeGP" width="70%" />
  

  
## Experimental Results

#### Confusion Matrix
<img src="https://drive.google.com/uc?export=view&id=17riSuoWBL10PElMVecLOzLQTo73h6z63" width="50%" />

#### Key Metrics on Test Set
<img src="https://drive.google.com/uc?export=view&id=1_DN9zLf4VpAIzwdxgRbh85newq5xutQx" width="70%" />

#### Benchmark Comparison - Sensitivity and Precision 
<img src="https://drive.google.com/uc?export=view&id=1yn8bp9FqNpxShQkXxGbiC7R3zub4_W1C" width="70%" />
<img src="https://drive.google.com/uc?export=view&id=1OiaM3UwCr_SFL0NKBuWMU62-Aa21Omcx" width="70%" />
