# Variationally Regularized Graph-based Representation Learning for Electronic Health Records

## Introduction
This project aims to tackle the challenge of optimizing representation learning from Electronic Health Records (EHR) in the healthcare domain.  We propose an innovative approach employing Graph Neural Networks (GNN) to learn these interconnections in EHR data. Our heterogeneous graph network exhibits robustness in understanding graph structures, leading to adaptive performance enhancements across various predictive tasks in EHR. 
Currently, we are training the baseline model named VGNN and plan to extend our dataset by creating multiple HADM IDs for each patient and aggregating lab-event features. We are also going to build our heterogeneous graph neural network.


## Data Preprocessing
To train VGNN on our dataset, we have revised the preprocessing code. The code extracts train/valid/test datasets from raw MiMic3 csvs (`../../rawdata/mimic/`). Use the following command:
```
python3 preprocess_mimic.py --input_path ../../rawdata/mimic/
```
We have made the following revisions:
* Set the path to read from our datasets and match the data type.
* Make it optional to use lab-event features.
* Map one patient to one HADM ID.


## BaselineModel VGNN Training
We trained VGNN [Variationally Regularized Graph-based Representation Learning for Electronic Health Records](https://arxiv.org/abs/1912.03761) for mortality prediction using our dataset. The figure shows the archtecture of VGNN.
<img src="https://github.com/NYUMedML/GNN_for_EHR/blob/master/plots/model.png" alt="drawing" width="600"/>


### Train
VGNN can be trained by running the command:
```
python3 train.py --data_path ./data/mimic/
```
We trained VGNN with different dropout rates, and the results of training can be found in the following directories:
* `lr_0.0001-input_256-output_256-dropout_0.2/`
* `lr_0.0001-input_256-output_256-dropout_0.4/`
* `lr_0.0001-input_256-output_256-dropout_0.5/`
Additionally, we added codes for evaluating the baseline by AUPRC on the test dataset.


### Environment
Required packages can be installed on **python3.6** environment via command:
```
pip3 install -r requirements.txt
```
We revise `requirements.txt`  for our specific environment with an Nvidia GPU and Cuda 11.7.