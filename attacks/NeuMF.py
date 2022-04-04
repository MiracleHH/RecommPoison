import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuMF(nn.Module):
    """docstring for NeuMF"""
    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NeuMF, self).__init__()
        self.mf_embedding_user=nn.Embedding(num_users,mf_dim)
        self.mf_embedding_item=nn.Embedding(num_items,mf_dim)
        self.mlp_embedding_user=nn.Embedding(num_users,layers[0]//2)
        self.mlp_embedding_item=nn.Embedding(num_items,layers[0]//2)
        self.mlp_layers=nn.ModuleList()
        for idx in range(1,len(layers)):
            self.mlp_layers.append(nn.Linear(layers[idx-1],layers[idx]))
        self.affine_output = nn.Linear(in_features=layers[-1]+mf_dim, out_features=1)
        self.weight_init()

    def forward(self, user,item):
        mf_user_embeddings=self.mf_embedding_user(user)
        mf_item_embeddings=self.mf_embedding_item(item)

        mlp_user_embeddings=self.mlp_embedding_user(user)
        mlp_item_embeddings=self.mlp_embedding_item(item)

        gmf=mf_user_embeddings.mul(mf_item_embeddings)
        mlp=torch.cat((mlp_user_embeddings,mlp_item_embeddings),-1)
        num_layer = len(self.mlp_layers)
        for idx in range(num_layer):
            mlp=F.relu(self.mlp_layers[idx](mlp))

        predict=torch.cat((gmf,mlp),-1)
        predict=torch.sigmoid(self.affine_output(predict))

        return predict

    def forward_with_weights(self,user,item,weights):
        mf_user_embeddings=F.embedding(user,weights[0])
        mf_item_embeddings=F.embedding(item,weights[1])

        mlp_user_embeddings=F.embedding(user,weights[2])
        mlp_item_embeddings=F.embedding(item,weights[3])

        gmf=mf_user_embeddings.mul(mf_item_embeddings)
        mlp=torch.cat((mlp_user_embeddings,mlp_item_embeddings),-1)

        num_layer = len(self.mlp_layers)
        for idx in range(num_layer):
            mlp=F.relu(F.linear(mlp,weights[4+idx*2],weights[5+idx*2]))

        predict=torch.cat((gmf,mlp),-1)
        predict=torch.sigmoid(F.linear(predict,weights[-2],weights[-1]))

        return predict


    def weight_init(self):
        nn.init.normal_(self.mf_embedding_user.weight,mean=0.0,std=0.01)
        nn.init.normal_(self.mf_embedding_item.weight,mean=0.0,std=0.01)
        nn.init.normal_(self.mlp_embedding_user.weight,mean=0.0,std=0.01)
        nn.init.normal_(self.mlp_embedding_item.weight,mean=0.0,std=0.01)



def train_on_batch(model,batch_user,batch_item,batch_label,optimizer,criterion,device):
    optimizer.zero_grad()
    batch_user,batch_item,batch_label=batch_user.to(device),batch_item.to(device),batch_label.to(device)
    outputs=model(batch_user,batch_item)
    loss=criterion(outputs.view(-1),batch_label)
    loss.backward()
    optimizer.step()
    return loss.item()