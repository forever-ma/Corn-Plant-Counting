# Corn-Plant Counting Using Scare-Aware Feature and Channel Interdependence
This is a pytorch implemention of paper "Corn-Plant Counting Using Scare-Aware Feature and Channel Interdependence".

Prerequisites
===
We recommend Anaconda as the environment. <br>

Python: 3.6 or later <br>

Pytorch: 1.0.0 or later <br>


Installation
===
1. Install visdom <br>
```
pip install visdom
```
2. Install tqdm <br>
```
pip install tqdm
```
3.Clone this repository
```
git clone https://github.com/forever-ma/Corn-Plant-Counting.git
```


Preprocessing
===
1. You can download Corn-Plant Dataset from [GoogleDrive](https://drive.google.com/file/d/1GF6HaDgInQ89OrHR0tPRzrOjeF44micy/view?usp=sharing) <br>

2. Generation the ground-truth density maps for training and testing <br>
```
python data_preparation/same_gaussian_kernel.py
````

Training
===
1. Modify the root path in "train.py" according to your dataset position. <br>

2. Using the following command:
```
python -m visdom.server
```

3. the command line for training is as follows:
```
python train.py
```

Testing
===
1. Modify the root path in "test.py" according to your dataset position. <br>

2. Using the following command for calculating MAE and MSE of test images or just show an estimated density-map:
```
python test.py
```

Model
===
We got the best MAE and MSE at the 596 epoch for the experiment under mild illumination intensity. You can download this epoch's model from [Link](https://drive.google.com/file/d/1CL3o5K125Lb-hPG31qvKMAllDN4qOINf/view?usp=sharing).
