import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import os
from data import SampleGenerator, UserItemRatingDataset
import argparse
from time import time
import random
from evaluate import evaluate_model,get_single_HR
from NeuMF import NeuMF, train_on_batch
os.environ['CUDA_VISIBLE_DEVICES']='0'
seed_value=42


def parse_args():
    """Arguments"""
    parser = argparse.ArgumentParser(description="Run Bandwagon Attack.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. \
                        So layers[0]/2 is the embedding size.")
    parser.add_argument('--l2_reg', type=float, default=0,
                        help='L2 Regularization for the optimizer.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--m', type=int, default=2,
                        help='The number of fake users.')
    parser.add_argument('--n', type=int, default=30,
                        help='The maximum number of filler items for one fake user.')
    parser.add_argument('--topK', type=int, default=10,
                        help='The maximum number of items in a user\'s recommendation list.')
    parser.add_argument('--targetItem', type=int, default=1072,
                        help='The target item ID.')
    return parser.parse_args()


if __name__ == '__main__':
    t1 = time()
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    l2_reg = args.l2_reg
    num_negatives = args.num_neg
    learning_rate = args.lr
    m = args.m
    num_fillers = args.n
    topK = args.topK
    target_item = args.targetItem

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Bandwagon attack arguments: %s " % (args))

    start=time()
    # Load Data
    data_dir = args.path + args.dataset

    sample_generator = SampleGenerator(path=data_dir)
    num_users,num_items=sample_generator.get_size() 
    user_input, item_input, labels = sample_generator.instance_train_data(num_negatives,seed_value)
    evaluate_data=sample_generator.get_evaluate_data()
    normal_users = sample_generator.get_normal_users(num_users, target_item)
    negative_items = sample_generator.get_negative_items()
    candidate_items = sample_generator.get_popular_items()

    print('num_users: {}, num_items: {}'.format(num_users,num_items))

    np.random.seed(seed_value)
    seeds_datasets=np.random.randint(100000,size=2*num_users)

    split = 0.1
    pos = int(round(num_items * split))
    num_popular= int(round(num_fillers * split))
    num_normal = num_fillers - num_popular

    filler_items=[]

    for i in range(m):
        candidates=candidate_items.copy()
        candidates.remove(target_item)

        temp=[]

        # Selected popular items
        popular_indices=random.sample(range(pos),num_popular)
        for idx in popular_indices:
            temp.append(candidates[idx])
        for j in range(1,num_popular+1):
            candidates.remove(temp[-j])

        # Selected normal items
        normal_indices = random.sample(range(len(candidates)), num_normal)
        for idx in normal_indices:
            temp.append(candidates[idx])
        for j in range(1,num_normal+1):
            candidates.remove(temp[-j])

        temp.append(target_item)
        filler_items.append(list(temp))
        print('Poison #{} has been generated!'.format(i))

        # Generate negative instances
        np.random.seed(seeds_datasets[num_users+i])
        negative_indices=np.random.randint(len(candidates),size=(num_fillers+1)*num_negatives)
        temp.extend([candidates[idx] for idx in negative_indices])

        poison_labels=[1]*(num_fillers+1)
        poison_labels.extend([0]*(num_fillers+1)*num_negatives)

        user_input.extend([num_users+i]*(num_fillers+1)*(num_negatives+1))
        item_input.extend(temp)
        labels.extend(poison_labels)

    print('filler items:\n{}'.format(filler_items))
    isExists=os.path.exists('poison')
    if not isExists:
        os.makedirs('poison')
    with open("poison/poison_{}_bandwagon_{}_{}_{}.txt".format(args.dataset,target_item,m,num_fillers),"w") as f:
        f.writelines(str(filler_items))

    torch_dataset=UserItemRatingDataset(user_tensor=torch.LongTensor(user_input),
        item_tensor=torch.LongTensor(item_input),
        target_tensor=torch.FloatTensor(labels))
    loader=Data.DataLoader(dataset=torch_dataset,batch_size=batch_size,shuffle=True,num_workers=2)

    criterion=nn.BCELoss()

    res=[]
    for i in range(30):
        model=NeuMF(num_users+m, num_items, mf_dim, layers).to(device)
        optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=l2_reg)
        
        print('Start training model #{}'.format(i))
        model.train()
        best_hr, best_ndcg, best_iter=-1,-1,-1
        for epoch in range(num_epochs):
            t1=time()
            total_loss=0.0
            for batch_user,batch_item,batch_label in loader:
                total_loss+=train_on_batch(model,batch_user,batch_item,batch_label,optimizer,criterion,device)
            t2=time()
            hr, ndcg = evaluate_model(model, evaluate_data, topK,device)
            print('Epoch {} [{:.2f} s]: HR = {:.4f}, NDCG = {:.4f}, loss = {:.4f} [{:.2f} s]'.format(epoch,t2-t1,hr,ndcg,total_loss,time()-t2))
            if best_hr<hr:
                torch.save(model.state_dict(), 'Pretrain/pytorch_bandwagon_model.tar')
                best_hr,best_ndcg,best_iter=hr,ndcg,epoch
        
        print('End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. '.format(best_iter, best_hr, best_ndcg))
        model.load_state_dict(torch.load('Pretrain/pytorch_bandwagon_model.tar'))
        model.eval()

        hr=get_single_HR(model,num_users,normal_users,negative_items,target_item,topK,device)
        print('HR@{} for target item {}: {}'.format(topK,target_item,hr))
        print('-'*80)
        res.append(hr)
        del model

    print('\nAll HR@{}:'.format(topK))
    for r in res:
        print(r)

    print('\nAverage HR@{}: {}'.format(topK,np.array(res).mean()))
    print('\nm={}, n={}, targetItem={}, epochs={}'.format(m,num_fillers,target_item,num_epochs))
    print('Time:{:.2f}s'.format(time()-start))
