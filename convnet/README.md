# A simple CNN architecture from scratch
## Python and dependencies
- Python 3
- environment.yml contains a list of libraries needed to set environment, You can use it to create a copy of conda environment
- $ conda env create -f environment.yml

## Data Loading
- Data loading is the very first step of any machine learning pipelines. First, you should download the CIFAR-10  dataset with our provided script under ./data by:
- $ sh get_data . sh
- The script downloads data to the ./data folder.

##train a small CNN with only 50 samples in CIFAR-10 dataset. 
-  $ python train.py
