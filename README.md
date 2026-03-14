# Deep Learning for Time Series: LSST Classification

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20TensorFlow-orange.svg)

##  Project Description
This repository contains the source code for our evaluation project for the **"Deep Learning for Time Series"** course. 

The main objective is to perform a **Time Series Classification (TSC)** task on the multivariate astronomical dataset **LSST** (Large Synoptic Survey Telescope).

In accordance with the project guidelines, we chose **Setting 2**:
1. **Pre-training (Forecasting)**: Training an encoder model on a forecasting task using exclusively the *Informer* family datasets (ETT, Electricity, Weather, etc.).
2. **Adaptation (Classification)**: Transferring the pre-trained encoder and fine-tuning it on the LSST dataset for the 14-class classification task.

## Repository structure 

```text 
LSST-TimeSeries-DL/
│
├── data/
│   ├── lsst/
│   └── informer/
│
├── src/                       
│   ├── __init__.py           
│   ├── dataset.py           
│   ├── model.py               
│   └── preprocessing.py       
│
├── baseline.py                
├── pretrain.py               
├── finetune.py             
│
├── pretrained_encoder.pth      
├── requirements.txt            
├── report_DL_TimeSeries.pdf    
└── README.md