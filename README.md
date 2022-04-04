# Data Poisoning Attacks to Deep Learning Based Recommender Systems
This repository is the major code implementation for the paper:

Hai Huang, Jiaming Mu, Neil Zhenqiang Gong, Qi Li, Bin Liu, Mingwei Xu. [Data Poisoning Attacks to Deep Learning Based Recommender Systems](https://www.ndss-symposium.org/wp-content/uploads/2021-525-paper.pdf). In ISOC Network and Distributed System Security Symposium (NDSS), 2021.

## Setup
### Environment requirements

Python ≥ 3.6, Pandas ≥ 1.0, NumPy ≥ 1.19, PyTorch ≥ 1.4.

### Dataset Preparation

Please follow the instructions in [data processing](./data_processing/) to prepare the datasets for experiments.

## Examples

Before conducting attacks, you should firstly copy the processed dataset to the `./attacks/Data` folder, and then execute the following commands under the `./attacks` folder.

Take the [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) dataset for example, you can use the following command to conduct our attack:

```Shell
python main.py --path Data/ --dataset ml-100k --epochs 20 --batch_size 256 --num_factors 8 --layers "[64,32,16,8]" --l2_reg 0 --num_neg 4 --lr 0.001 --alpha 1E-2 --kappa 1.0 --rounds 5 --m 5 --n 30 --topK 10 --targetItem 100 --reg_u 1E2 --prob 0.9 --s 1
```

You can use the following command to conduct the random attack:

```Shell
python random_attack.py --path Data/ --dataset ml-100k --epochs 20 --batch_size 256 --num_factors 8 --layers "[64,32,16,8]" --l2_reg 0 --num_neg 4 --lr 0.001 --m 5 --n 30 --topK 10 --targetItem 100
```

Similarly, you can use the following command to conduct the bandwagon attack:

```Shell
python bandwagon_attack.py --path Data/ --dataset ml-100k --epochs 20 --batch_size 256 --num_factors 8 --layers "[64,32,16,8]" --l2_reg 0 --num_neg 4 --lr 0.001 --m 5 --n 30 --topK 10 --targetItem 100
```

You can refer to the instructions in [attacks](./attacks/) for more details.


## Citation

Please cite our paper if you use this code in your own research work:

```
@inproceedings{HMGLLX21,
author = {Hai Huang and Jiaming Mu and Neil Zhenqiang Gong and Qi Li and Bin Liu and Mingwei Xu},
title = {{Data Poisoning Attacks to Deep Learning Based Recommender Systems}},
booktitle = {{Network and Distributed System Security Symposium (NDSS)}},
publisher = {Internet Society},
year = {2021}
}
```

## Acknowledgements

We thank the maintainers of the following open-sourced repositories:

1. [https://github.com/hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)
2. [https://github.com/yihong-chen/neural-collaborative-filtering](https://github.com/yihong-chen/neural-collaborative-filtering)
3. [https://github.com/fuying-wang/Data-poisoning-attacks-on-factorization-based-collaborative-filtering](https://github.com/fuying-wang/Data-poisoning-attacks-on-factorization-based-collaborative-filtering)