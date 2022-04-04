import pandas as pd
from data import SampleGenerator
from time import time
import argparse
import numpy as np
import csv


def parse_args():
    """Arguments"""
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/ml-100k',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--sep', nargs='?', default='\t',
                        help='The seperator for each line in dataset.')
    parser.add_argument('--header', type=int, default=None,
                        help='Row number(s) to use as the column names, and the start of the data.')
    return parser.parse_args()


if __name__ == '__main__':
    t1 = time()
    args = parse_args()
    num_negatives = args.num_neg

    print("Dataset processing arguments: %s " % (args))

    start=time()
    # Load Data
    data_dir = args.path + '/'+args.dataset
    
    data_rating = pd.read_csv(data_dir, sep=args.sep, header=args.header, names=['userId', 'itemId', 'rating', 'timestamp'],
                              engine='python')

    print('Range of userId is [{}, {}]'.format(data_rating.userId.min(), data_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(data_rating.itemId.min(), data_rating.itemId.max()))
    

    sample_generator = SampleGenerator(ratings=data_rating)
    training_ratings,evaluate_ratings=sample_generator.get_data_split()
    evaluate_data = sample_generator.get_evaluate_data()

    columns=['userId','itemId']
    columns.extend(['negativeId_{}'.format(x) for x in range(99)])
    evaluate_negative_samples=pd.DataFrame(columns=columns)

    for row in evaluate_data.itertuples():
        _tmp = []
        _tmp.append(int(row.userId))
        _tmp.append(int(row.itemId))
        for i in range(len(row.negative_samples)):
            _tmp.append(int(row.negative_samples[i]))
        evaluate_negative_samples=evaluate_negative_samples.append({columns[i]:_tmp[i] for i in range(101)},ignore_index=True)

    training_ratings.to_csv(args.path+'/train.csv',index=None)
    evaluate_ratings.to_csv(args.path+'/evaluate.csv',index=None)
    evaluate_negative_samples.to_csv(args.path+'/evaluate_negative_samples.csv',index=None)

    statistics=training_ratings.itemId.value_counts()
    statistics.to_csv(args.path+'/statistics.csv')

    print('Finished!')