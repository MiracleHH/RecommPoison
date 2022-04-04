import random
import pandas as pd
from copy import deepcopy
import numpy as np
import torch
import torch.utils.data as Data


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, path):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        train_ratings = pd.read_csv(path+'/train.csv')
        self.evaluate_negative_samples=pd.read_csv(path+'/evaluate_negative_samples.csv')
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.train_ratings = self._binarize(train_ratings)
        self.num_users,self.num_items=self.train_ratings.userId.max()+1,max(self.train_ratings.itemId.max(),self.evaluate_negative_samples.itemId.max())+1
        self.item_pool = set(list(range(self.num_items)))
        self.negatives = self._sample_negative(train_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'] = ratings.rating.apply(lambda x: 1.0 if x > 0 else 0.0)
        return ratings

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        return interact_status[['userId', 'interacted_items','negative_items']]

    def instance_train_data(self, num_negatives,seed=None):
        """instance training data"""
        num_users=len(self.negatives)
        np.random.seed(seed)
        seeds=np.random.randint(100000,size=2*num_users)
        users, items, ratings = [], [], []
        for i in range(num_users):
            interacted_items=list(self.negatives.iloc[i]['interacted_items'])
            users.extend([self.negatives.iloc[i]['userId']]*len(interacted_items)*(num_negatives+1))
            items.extend(interacted_items)
            negative_items=list(self.negatives.iloc[i]['negative_items'])
            np.random.seed(seeds[i])
            indices=np.random.randint(len(negative_items),size=len(interacted_items)*num_negatives)
            items.extend([negative_items[idx] for idx in indices])
            ratings.extend([1.0]*len(interacted_items))
            ratings.extend([0.0]*len(interacted_items)*num_negatives)
        return users, items, ratings

    def get_normal_users(self, num_users, target_item):
        """get normal users who haven't rated the target item"""
        rating_users = \
            self.train_ratings[(self.train_ratings['itemId'] == target_item) & (self.train_ratings['rating'] == 1.0)][
                'userId'].drop_duplicates()
        temp = list(range(num_users))
        for r in rating_users:
            temp.remove(r)
        return temp

    def get_evaluate_data(self):
        """create evaluate data"""
        return np.array(self.evaluate_negative_samples)

    def get_negative_items(self):
        """negative items for all users"""
        negative_items = []
        negative_ratings = self.negatives[['userId', 'negative_items']]
        for row in negative_ratings.itertuples():
            negative_items.append(list(row.negative_items))
        return negative_items

    def get_popular_items(self):
        """get popular items with high frequencies"""
        res = self.train_ratings.itemId.value_counts()
        return list(res.index)

    def get_size(self):
        return self.num_users,self.num_items


class UserItemRatingDataset(Data.Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

