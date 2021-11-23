# A pipeline of training neural networks to recognize MNIST Handwritten Digits.

## Python and dependencies
- Python 3
- environment.yml contains a list of libraries needed to set environment, You can use it to create a copy of conda environment
- $ conda env create -f environment.yml

## Data Loading
- Data loading is the very first step of any machine learning pipelines. First, you should download the MNIST dataset with our provided script under ./data by:
- $ sh get_data . sh
- The script downloads MNIST data (mnist_train.csv and mnist_test.csv) to the ./data folder.

## The main.py contains the major logic of this code. You can execute it by invoking the following command where the yaml file contains all the hyper-parameters.
-  $ python main.py --config configs/<name_of_config_file>.yaml
