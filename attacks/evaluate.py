'''
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
'''
import torch
import math
import numpy as np
from time import time


def evaluate_model(model, evaluate_data, K,device):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """

    hits, ndcgs = [], []
    for row in evaluate_data:
        hr, ndcg = eval_one_rating(model, row[0],row[1:],K,device)
        hits.append(hr)
        ndcgs.append(ndcg)
    return np.array(hits).mean(), np.array(ndcgs).mean()


def eval_one_rating(model, user, negative_items, top_k,device):
    item=negative_items[0]
    users = np.full(negative_items.size, user)

    with torch.no_grad():
        users=torch.LongTensor(users)
        negative_items=torch.LongTensor(negative_items)
        users,negative_items=users.to(device),negative_items.to(device)
        ratings = model(users,negative_items)
        ratings,indices=torch.topk(ratings.view(-1),top_k,dim=0,largest=True)
        ranklist = [negative_items[i].item() for i in indices]
        hr = getHitRatio(ranklist, item)
        ndcg = getNDCG(ranklist, item)
        return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


# calculate Hit Ratio for one specific target item
def get_single_HR(model,num_users,normal_users,negative_reference,target_item,topK,device):
    count=0.0
    with torch.no_grad():
        for user in normal_users:
            ratings=model(torch.LongTensor([user]*len(negative_reference[user])).to(device),torch.LongTensor(negative_reference[user]).to(device))
            ratings,indices=torch.topk(ratings.view(-1),topK,dim=0,largest=True)
            recommendation=[negative_reference[user][idx] for idx in indices]
            if target_item in recommendation:
                count+=1
    return count/num_users