# CVPR-24

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:
- torch==2.0.1
- torchvision==0.15.2
- numpy==1.24.4
- matplotlib==3.7.3
- pillow==10.0.1
- pip==23.2.1
- python==3.8.18

## Run pipeline for Hier-FUN
1. Entering the Hier-FUN
```python
cd CVPR-24
cd Hier-FUN
```

2. Specify the dataset path in fl_datasets.py, e.g., 
```python
dataset_path = r'xxx/zgma/dataset'
```

3. Specify the file path in start_hier-fun.py, e.g,
```python
source_code_path = 'xxx/zgma/Hier-FUN/'
exp_results_path = 'xxx/Hier-FUN/results/'
python_path = 'xxx/miniconda/envs/zgma/bin/python3'
```

4. Change the experimental settings as you need, e.g, to run MLP over fmnist with 21 devices, you can set
```python
method=0 # for FedAvg
ratio=0.9 # under level 1
dataset='fmnist' # MLP over fmnist
num_clients=21 # with 21 devices
```
Then, you can run
```python
python start_hier-fun.py
```
