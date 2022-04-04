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
    parser = argparse.ArgumentParser(description="Run Our Attack.")
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
    parser.add_argument('--reg_u', type=float, default=0,
                        help='Regularization for poisoned labels.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=100.0,
                        help='The coefficient used to adjust the weight of loss functions')
    parser.add_argument('--kappa', type=float, default=0.0,
                        help='The coefficient used to enhance the roubustness of attacks')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of rounds to update poison model.')
    parser.add_argument('--m', type=int, default=2,
                        help='The number of fake users.')
    parser.add_argument('--n', type=int, default=30,
                        help='The maximum number of filler items for one fake user.')
    parser.add_argument('--topK', type=int, default=10,
                        help='The maximum number of items in a user\'s recommendation list.')
    parser.add_argument('--targetItem', type=int, default=100,
                        help='The target item ID.')
    parser.add_argument('--prob', type=float, default=1.0,
                        help='Probability attenuation coefficient.')
    parser.add_argument('--s', type=int, default=1,
                        help='Step size for poisoning.')
    return parser.parse_args()


def target_loss(x, y,kappa):
    eps=1E-12
    loss=torch.log(x+eps)-torch.log(y+eps)
    if loss>-kappa:
        return loss
    return -kappa

def get_f(model,target_users,unrated_items,candidate_users,candidate_items,alpha,kappa,reg_u,target_item,topK,optimizer,device):
    total_loss=0.0

    target_users=np.array(target_users)
    np.random.shuffle(target_users)

    for user in target_users:
        items=torch.LongTensor(unrated_items[user])
        users=torch.LongTensor([user]*len(items))
        users,items=users.to(device),items.to(device)

        ratings=model(users,items)
        _ratings,indices=torch.topk(ratings.view(-1),topK,dim=0,largest=True)

        base_score=model(torch.LongTensor([user]).to(device),torch.LongTensor([target_item]).to(device))[0][0]
        loss=target_loss(_ratings[-1],base_score,kappa)
        if loss==-kappa:
            total_loss+=alpha*reg_u*loss
            continue
        loss=alpha*reg_u*loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss
    
    label_poisoned=model(torch.LongTensor(candidate_users).to(device),torch.LongTensor(candidate_items).to(device)).view(-1)
    loss=alpha*(label_poisoned**2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss+=loss
    
    return total_loss


if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    l2_reg = args.l2_reg
    reg_u=args.reg_u
    num_negatives = args.num_neg
    learning_rate = args.lr
    alpha=args.alpha
    kappa=args.kappa
    total_round=args.rounds
    m=args.m
    n=args.n
    topK=args.topK
    target_item=args.targetItem
    s=args.s

    print("NeuMF arguments: %s " %(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start=time()
    # Load Data
    data_dir = args.path + args.dataset

    sample_generator = SampleGenerator(path=data_dir)
    num_users,num_items=sample_generator.get_size() 
    user_input, item_input, labels = sample_generator.instance_train_data(num_negatives,seed_value)
    evaluate_data=sample_generator.get_evaluate_data()
    target_users = sample_generator.get_normal_users(num_users, target_item)
    unrated_items = sample_generator.get_negative_items()

    print('num_users: {}, num_items: {}'.format(num_users,num_items))

    np.random.seed(seed_value)
    seeds_datasets=np.random.randint(100000,size=2*num_users)


    criterion=nn.BCELoss()

    # You can also load previous data of filler items from external files to save time here
    filler_items=[]


    for i,items in enumerate(filler_items):
        poison_users=[num_users+i]*((n+1)*(num_negatives+1))
        poison_labels=[1]*(n+1)
        poison_labels.extend([0]*(n+1)*num_negatives)
        poison_items=list(items)
        t=list(range(num_items))
        for j in items:
            t.remove(j)
        np.random.seed(seeds_datasets[num_users+i])
        negatives_indices=np.random.randint(len(t),size=(n+1)*num_negatives)
        for x in negatives_indices:
            poison_items.append(t[x])

        user_input.extend(poison_users)
        item_input.extend(poison_items)
        labels.extend(poison_labels)


    prob=torch.ones(num_items-1).to(device)
    m0=len(filler_items)

    steps=[s for i in range((m-m0)//s)]
    if (m-m0)%s!=0:
        steps.append((m-m0)%s)

    for i,step in enumerate(steps):
        if step==1:
            print('Start generating poison #{} !'.format(m0+i*s))
        else:
            print('Start generating poison #{}-{} !'.format(m0+i*s,m0+i*s+step-1))
        print('-' * 80)

        candidate_users,candidate_items=[],[]

        for j in range(step):
            candidate_users.extend([num_users+m0+i*s+j]*(num_items-1))
            tmp=list(range(num_items))
            tmp.remove(target_item)
            candidate_items.extend(tmp)
            user_input.append(num_users+m0+i*s+j)
            item_input.append(target_item)
            labels.append(1)

        poison_model=NeuMF(num_users+m0+s*i+step, num_items, mf_dim, layers).to(device)
        optimizer=optim.Adam(poison_model.parameters(),lr=learning_rate,weight_decay=l2_reg)


        torch_dataset=UserItemRatingDataset(user_tensor=torch.LongTensor(user_input),
            item_tensor=torch.LongTensor(item_input),
            target_tensor=torch.FloatTensor(labels))
        loader=Data.DataLoader(dataset=torch_dataset,batch_size=batch_size,shuffle=True,num_workers=2)

        
        poison_model.train()
        best_hr, best_ndcg, best_iter=-1,-1,-1
        for epoch in range(num_epochs):
            t1=time()
            total_loss=0.0
            for batch_user,batch_item,batch_label in loader:
                total_loss+=train_on_batch(poison_model,batch_user,batch_item,batch_label,optimizer,criterion,device)
            t2=time()
            hr, ndcg = evaluate_model(poison_model, evaluate_data, topK,device)
            print('Epoch {} [{:.1f}s]: HR = {:.4f}, NDCG = {:.4f}, loss = {:.4f} [{:.1f}s]'.format(epoch,t2-t1,hr,ndcg,total_loss,time()-t2))
            if best_hr<hr:
                torch.save(poison_model.state_dict(), 'Pretrain/pytorch_tmp_poison_model.tar')
                best_hr,best_ndcg,best_iter=hr,ndcg,epoch
        
        poison_model.load_state_dict(torch.load('Pretrain/pytorch_tmp_poison_model.tar'))
        torch.save(poison_model.state_dict(), 'Pretrain/pytorch_modified_poison_model.tar')
        poison_model.eval()

        target_hr=get_single_HR(poison_model,num_users,target_users,unrated_items,target_item,topK,device)
        print('Initial target HR: {:.4f}, best HR: {:.4f}'.format(target_hr,best_hr))

        best_hr=target_hr
        stop=False
        poison_model.train()
        for epoch in range(total_round):
            t1=time()
            loss1=get_f(poison_model,target_users,unrated_items,candidate_users,candidate_items,alpha,kappa,reg_u,target_item,topK,optimizer,device)

            loss2=0.0
            for batch_user,batch_item,batch_label in loader:
                loss2+=train_on_batch(poison_model,batch_user,batch_item,batch_label,optimizer,criterion,device)
            t2=time()
            hr, ndcg = evaluate_model(poison_model, evaluate_data, topK,device)
            print('Epoch {} [{:.1f}s]: HR = {:.4f}, NDCG = {:.4f}, loss1 = {:.4f}, loss2 = {:.4f} [{:.1f}s]'.format(epoch,t2-t1,hr,ndcg,loss1,loss2,time()-t2))
            t2=time()
            target_hr=get_single_HR(poison_model,num_users,target_users,unrated_items,target_item,topK,device)
            print('Target HR: {:.4f} [{:.1f}s]'.format(target_hr,time()-t2))
            
            if target_hr>best_hr:
                best_hr=target_hr
                torch.save(poison_model.state_dict(), 'Pretrain/pytorch_modified_poison_model.tar')
            

        poison_model.load_state_dict(torch.load('Pretrain/pytorch_modified_poison_model.tar'))
        poison_model.eval()

        
        with torch.no_grad():
            ratings=poison_model(torch.LongTensor(candidate_users).to(device),torch.LongTensor(candidate_items).to(device)).view(step,-1)

            for j in range(step):
                _ratings=ratings[j].mul(prob)
                _ratings,_indices=torch.topk(_ratings,n,dim=-1,largest=True)
                poison_items=[candidate_items[:num_items-1][p] for p in _indices]

                prob[_indices]*=args.prob
                if (prob<1.0).all():
                    prob=torch.ones(num_items-1).to(device)

                poison_items.append(target_item)
                print('Poison items:\n{}'.format(poison_items))
                filler_items.append(list(poison_items))

                tmp=list(range(num_items))
                for t in poison_items:
                    tmp.remove(t)
                np.random.seed(seeds_datasets[num_users+m0+i*s+j])
                negatives_indices=np.random.randint(len(tmp),size=(n+1)*num_negatives)
                poison_items.remove(target_item)
                for x in negatives_indices:
                    poison_items.append(tmp[x])

                user_input.extend([num_users+m0+i*s+j]*((n+1)*(num_negatives+1)-1))
                item_input.extend(poison_items)
                labels.extend([1]*n)
                labels.extend([0]*((n+1)*num_negatives))

        if step==1:
            print('Poison #{} has been generated!\n'.format(m0+i*s))
        else:
            print('Poison #{}-{} have been generated!\n'.format(m0+i*s,m0+i*s+step-1))
        del poison_model


    torch_dataset=UserItemRatingDataset(user_tensor=torch.LongTensor(user_input),
        item_tensor=torch.LongTensor(item_input),
        target_tensor=torch.FloatTensor(labels))
    loader=Data.DataLoader(dataset=torch_dataset,batch_size=batch_size,shuffle=True,num_workers=2)

    print('Filler items:\n{}'.format(filler_items))
    isExists=os.path.exists('poison')
    if not isExists:
        os.makedirs('poison')
    with open("poison/poison_{}_our_attack_{}_{}_{}.txt".format(args.dataset,target_item,m,n),"w") as f:
        f.writelines(str(filler_items))


    # The final evaluation
    res=[]
    for i in range(30):
        poison_model=NeuMF(num_users+m, num_items, mf_dim, layers).to(device)
        optimizer=optim.Adam(poison_model.parameters(),lr=learning_rate,weight_decay=l2_reg)
        
        print('Start training poison model #{}'.format(i))
        poison_model.train()
        best_hr, best_ndcg, best_iter=-1,-1,-1
        for epoch in range(num_epochs):
            t1=time()
            total_loss=0.0
            for batch_user,batch_item,batch_label in loader:
                total_loss+=train_on_batch(poison_model,batch_user,batch_item,batch_label,optimizer,criterion,device)
            t2=time()
            hr, ndcg = evaluate_model(poison_model, evaluate_data, topK,device)
            print('Epoch {} [{:.1f}s]: HR = {:.4f}, NDCG = {:.4f}, loss = {:.4f} [{:.1f}s]'.format(epoch,t2-t1,hr,ndcg,total_loss,time()-t2))
            if best_hr<hr:
                torch.save(poison_model.state_dict(), 'Pretrain/pytorch_poison_model.tar')
                best_hr,best_ndcg,best_iter=hr,ndcg,epoch
        
        print('End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. '.format(best_iter, best_hr, best_ndcg))
        poison_model.load_state_dict(torch.load('Pretrain/pytorch_poison_model.tar'))
        poison_model.eval()

        hr=get_single_HR(poison_model,num_users,target_users,unrated_items,target_item,topK,device)
        print('HR@{} for poison model #{}: {}'.format(topK,i,hr))
        print('-'*80)
        res.append(hr)
        del poison_model

    print('\nAll HR@{}:'.format(topK))
    for r in res:
        print(r)

    print('\nAverage HR@{}: {}'.format(topK,np.array(res).mean()))
    print('\nm={}, n={}, targetItem={}, epochs={}'.format(m,n,target_item,num_epochs))
    print('Time:{:.2f}s'.format(time()-start))
    
