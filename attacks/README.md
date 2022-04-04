# Data Poisoning Attacks to Deep Learning Based Recommender Systems
This repository is the major code implementation for the paper:

Hai Huang, Jiaming Mu, Neil Zhenqiang Gong, Qi Li, Bin Liu, Mingwei Xu. [Data Poisoning Attacks to Deep Learning Based Recommender Systems](https://www.ndss-symposium.org/wp-content/uploads/2021-525-paper.pdf). In ISOC Network and Distributed System Security Symposium (NDSS), 2021.

## Setup
### Environment requirements

Python ≥ 3.6, Pandas ≥ 1.0, NumPy ≥ 1.19, PyTorch ≥ 1.4.

## Quick Start

Take the MovieLens-100K dataset for example, you can use the following command to conduct our attack:

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

As for the MF attack, you can refer to the paper "Data poisoning Attack on Factorization-Based Collaborative Filtering" and its code implementation at [https://github.com/fuying-wang/Data-poisoning-attacks-on-factorization-based-collaborative-filtering](https://github.com/fuying-wang/Data-poisoning-attacks-on-factorization-based-collaborative-filtering) for more details.

Note that, the hit ratio for the whole model when evaluating the model under normal circumstances is **different** from the hit ratio for a single target item ( related to our attack goal ). You can refer to the NCF paper for more details of the former metric.

