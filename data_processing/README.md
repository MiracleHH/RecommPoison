# Dataset Processing
This is the code implementation for splitting the original dataset into a training dataset and a test dataset, and it follows the leave-one-out evaluation rule (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16).


## Setup
### Environment requirements

Python ≥ 3.6, Pandas ≥ 1.0, NumPy ≥ 1.19.

### Dataset requirements

Every dataset should contain four columns which represent `userId`, `itemId`, `rating` and `timestamp` respectively. Please re-arrange the IDs of users or items if there are some IDs (between the minimum ID and the maximum ID) never appearing in the original dataset to constitute a smaller and denser user-item interaction matrix before using this repository to split the dataset. Also, make sure that the IDs for both users and items start from 0.

## Quick Start

Take the [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) dataset for example, you can use the following command to process the dataset (Please follow the above requirements to convert the original `u.data` file to the `ratings.csv` file):

```Shell
python main.py --path Data/ml-100k --dataset ratings.csv --num_neg 4 --sep "," --header 0
```

After it is completed, you will get the training dataset `train.csv` and the test dataset `evaluate_negative_samples.csv` for the original dataset. You can then copy these files or the whole folder to your destination file path for conducting subsequent attacks.


