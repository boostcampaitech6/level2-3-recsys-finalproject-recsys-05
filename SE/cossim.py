import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pandas as pd
from sklearn.externals import joblib

import itertools



class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.combinations = [list(element) for element in itertools.combinations(range(5), 2)]
        self.combinations_len = len(self.combinations)

    def __getitem__(self, idx):
        df_idx = idx // self.combinations_len
        combination_idx = idx % self.combinations_len

        x = self.df.iloc[df_idx, self.combinations[combination_idx]]
        x = torch.tensor(x, dtype=torch.int32, device='cuda')

        y = self.df.iloc[df_idx, -1]
        y = torch.tensor(y, dtype=torch.float64, device='cuda')

        return x, y
        
    def __len__(self):
        return len(self.combinations) * len(self.df)
    


class SimilarityModel(nn.Module):
    def __init__(self, config):
        super(SimilarityModel, self).__init__()
        self.embedding = nn.Embedding(config['n_layers'], config['emb_size'])


    def forward(self, input):
        input = input.transpose_(0, 1)
        embedded = self.embedding(input)
        A = embedded[0]
        B = embedded[1]
        output = F.cosine_similarity(A, B, dim=1)     

        return output



tier = 'diamond'
df = pd.read_csv(f'../data/{tier}_matches.csv')

config = {}
config['n_layers'] = int(df.drop(columns='result').max().max() + 1)
config['emb_size'] = 3

dataset = MyDataset(df)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

model = SimilarityModel(config).to('cuda')

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fun = nn.CrossEntropyLoss()

model.train()
for x, y in tqdm(loader):
    optimizer.zero_grad()
    output = model(x)

    # print(output, y)
    loss = loss_fun(output, y)
    loss.backward()
    optimizer.step()

joblib.dump(model, 'cossim.joblib')