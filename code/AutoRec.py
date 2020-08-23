
"""
해당 코드는 pyTorch를 통해서 다음의 [논문](https://akmenon.github.io/papers/autorec/autorec-paper.pdf)을 구현한 코드입니다. 
"""

import pickle
import pandas as pd
import numpy as np
import os, sys, gc 

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable



import matplotlib.font_manager as fm

fontpath = 'C:/Users/User/Anaconda3/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9).get_name()

path = 'C:/Users/User/Documents/RecSys/ml-10m/ml-10M100K/'
train = pd.read_csv(path + "r1.train", header=None, names=['userId','movieId','rating','Timestamp'])
test = pd.read_csv(path + "r1.test", header=None, names=['userId','movieId','rating','Timestamp'])


def preprocessing(df):
    df['movieId'] = df['userId'].astype(str).apply(lambda x: x.split('::')[1])
    df['rating'] = df['userId'].astype(str).apply(lambda x: x.split('::')[2])
    df['Timestamp'] = df['userId'].astype(str).apply(lambda x: x.split('::')[3])
    df['userId'] = df['userId'].astype(str).apply(lambda x: x.split('::')[0])   
    del df['Timestamp']
    df = df.astype(np.float32)
    return df


train = preprocessing(train)
test = preprocessing(test)

movie = pd.concat([train, test], axis=0).reset_index(drop=True)

user2idx = {}
for i, l in enumerate(movie['userId'].unique()):
    user2idx[l] = i

movie2idx = {}
for i, l in enumerate(movie['movieId'].unique()):
    movie2idx[l] = i

idx2user = {i: user for user, i in user2idx.items()}
idx2movie = {i: item for item, i in movie2idx.items()}


n_users, n_items = len(idx2user), len(idx2movie)

movie_tr = movie.loc[0:train.shape[0]].reset_index(drop=True)
movie_te = movie.loc[train.shape[0]:].reset_index(drop=True)


class MovieLenseDataset(Dataset):
    """ MovieLense dataset."""
    # Initialize your data, download, etc.
    def __init__(self, data, user_based):  
        self.user_based = user_based
        useridx = data['useridx'] = data['userId'].apply(lambda x: user2idx[x]).values
        movieidx = data['movieidx'] = data['movieId'].apply(lambda x: movie2idx[x]).values
        rating = data['rating'].values
        
        if self.user_based:
            i = torch.LongTensor([useridx, movieidx])
            v = torch.FloatTensor(rating)
            self.df = torch.sparse.FloatTensor(i, v, torch.Size([n_users, n_items])).to_dense()
        else:
            i = torch.LongTensor([movieidx, useridx])
            v = torch.FloatTensor(rating)
            self.df = torch.sparse.FloatTensor(i, v, torch.Size([n_users, n_items])).to_dense()
            
    def __getitem__(self, index):
        return self.df[index]

    def __len__(self):
        return len(self.df)



train_loader = DataLoader(MovieLenseDataset(movie_tr, True), batch_size=512, shuffle=True)
test_loader = DataLoader(MovieLenseDataset(movie_te, True), batch_size=512, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self, num_users, num_movies, is_user_based=True):
        super().__init__() # 부모 클래스(torch.nn.Module)의 init을 불러옴   
        self.num_users = num_users
        self.num_movies = num_movies     
        self.hidden_dim = 500
        if is_user_based:
            # encoder_weight
            self.encode_w = nn.Linear(self.num_movies, self.hidden_dim, bias=True)
            self.decode_w = nn.Linear(self.hidden_dim, self.num_movies, bias=True)
            
        else:
            self.encode_w = nn.Linear(self.num_users, self.hidden_dim, bias=True)
            self.decode_w = nn.Linear(self.hidden_dim, self.num_users, bias=True)
            
        torch.nn.init.xavier_uniform_(self.encode_w.weight)
        torch.nn.init.xavier_uniform_(self.decode_w.weight)  
        
        self.encoder = nn.Sequential(
                        self.encode_w,
                        nn.Sigmoid()
                        )
        
        self.decoder = nn.Sequential(
                        self.decode_w,
                        nn.Identity()
                        )        
        
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        return self.decode(self.encode(x))


def MSEloss(inputs, targets):
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    
    # 해당 주석을 제거하면 값이 0~5사이로 자동적으로 매핑됨
    # 단, 실험결과 이렇게 하면 초반의 weight에 대한 학습이 잘 안되는 것 같음  
    # inputs = torch.clamp(inputs, min=0, max=5)
    criterion = nn.MSELoss(reduction='sum')
    return torch.sqrt(criterion(inputs * mask.float(), targets) / num_ratings)

dev = torch.cuda.set_device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = AutoEncoder(n_users, n_items, True).to(dev)


# weight_decay : L2 Regularization 
# item-based : weight_decay=0.01 , user-based : None 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00001)  # learning rate

def train():
    model.train()
    train_loss = 0
    for idx, train_batch in enumerate(train_loader):
        train_batch = train_batch.to(dev)
        optimizer.zero_grad()
        
        prediction = model(train_batch)
        loss = MSEloss(prediction, train_batch)    
        loss.backward()
        train_loss += loss.item() 
        optimizer.step()
    
    return train_loss / (idx+1)

def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, test_batch in enumerate(test_loader):
            test_batch = test_batch.to(dev)
            prediction = model(test_batch)
            loss = MSEloss(prediction, test_batch)
        
            test_loss += loss.item() 
            
    return test_loss / (idx+1)

from tqdm.notebook import tqdm as tqdm_notebook

nb_epochs = 100
train_losses = []
test_losses = []

for epoch in tqdm_notebook(range(0, nb_epochs)):
    train_loss = train()
    test_loss = test()
    if epoch % 1 == 0: 
        print('Epoch {:4d}/{} Train Loss: {:.6f} Test Loss: {:.6f}'.format(epoch+1, nb_epochs, train_loss, test_loss))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

import plotnine
from plotnine import *

loss_df_tr = pd.DataFrame()
loss_df_tr['RMSE'] = train_losses 
loss_df_tr['Epoch'] = loss_df_tr.index + 1
loss_df_tr['Type'] = 'Train'

loss_df_te = loss_df_tr.copy()
loss_df_te['RMSE'] = test_losses 
loss_df_te['Epoch'] = loss_df_te.index + 1
loss_df_te['Type'] = 'Test'

loss_df = pd.concat([loss_df_tr, loss_df_te], axis=0)
(ggplot(data=loss_df)
    + geom_line(aes(x='Epoch', y='RMSE', group='Type', color='Type'))
    + theme_minimal()
    + ggtitle("AutoRec Loss with PyTorch")
    + labs(x="Epoch", y="RMSE") 
    + theme(text = element_text(fontproperties=font),
         axis_text_x = element_text(angle=0, color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(12,6))
    + scale_color_hue(l=0.5)
 )


# ## 실험결과 
# - weight_decay에 따라서 Loss의 수렴하는 정도가 차이남 (작으면 작을 수록 점수상승이 큼)
# - 동일한 파라미터여도 초기값이 무엇으로 설정되었냐에 따라서 Local optimal (0.88)을 탈출하냐 못하냐가 결정됨 

