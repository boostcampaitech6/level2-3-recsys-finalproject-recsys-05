import torch
from torch import nn
import numpy as np



class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, USE_BIAS=True):
        super(MultiHeadAttention, self).__init__()
        if (cfg['hidden_size'] % cfg['n_head']) != 0:
            raise ValueError("d_feat(%d) should be divisible by b_head(%d)"%(cfg['hidden_size'], cfg['n_head']))
        self.d_feat = cfg['hidden_size']
        self.n_head = cfg['n_head']
        self.d_head = self.d_feat // self.n_head
        self.sq_d_k = np.sqrt(self.d_head)
        self.dropout = nn.Dropout(p=cfg['dropout'])

        self.lin_Q = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        self.lin_K = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        self.lin_V = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        

    def forward(self, input, position):
        n_batch = input.shape[0]
        
        pos_idx = position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.d_feat)
        input_Q = torch.gather(input, 2, pos_idx)

        Q = self.lin_Q(input_Q)
        K = self.lin_K(input)
        V = self.lin_V(input)

        Q = Q.view(n_batch, -1, self.n_head, self.d_head)
        K = K.view(n_batch, -1, self.n_head, self.d_head)
        V = V.view(n_batch, -1, self.n_head, self.d_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.sq_d_k 
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V) 

        output = output.transpose(1, 2).contiguous()
        output = output.view(n_batch, -1, self.d_feat)

        return output



class SASModel(nn.Module):
    def __init__(self, cfg):
        super(SASModel, self).__init__()
        ### matchId, summonerId, position은 제외한 categorical feature의 개수
        cate_used_len = len(cfg['cate_cols']) - 3

        # categorical
        self.cate_emb = nn.Embedding(cfg['n_layers'], cfg['emb_size'], padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg['emb_size'] * cate_used_len, cfg['hidden_size']),
            nn.LayerNorm(cfg['hidden_size']),
            nn.Dropout(p=cfg['dropout'])
        )

        # continuous
        self.cont_norm = nn.BatchNorm1d(len(cfg['cont_cols']))
        self.cont_proj = nn.Sequential(
            nn.Linear(len(cfg['cont_cols']), cfg['hidden_size']),
            nn.LayerNorm(cfg['hidden_size']),
            nn.Dropout(p=cfg['dropout'])
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(cfg['hidden_size'] * 2, cfg['hidden_size']),
            nn.LayerNorm(cfg['hidden_size']),
            nn.Dropout(p=cfg['dropout'])
        )

        self.self_attn = MultiHeadAttention(cfg)


    def forward(self, cate, cont, position):
        cate_size = [i for i in cate.size()]
        cate = self.cate_emb(cate).view(*cate_size[:-1], -1)
        cate = self.cate_proj(cate)
        
        cont_size = [i for i in cont.size()]
        cont = cont.view(-1, cont_size[-1])
        cont = self.cont_norm(cont).view(*cont_size)
        cont = self.cont_proj(cont)

        comb = torch.cat([cate, cont], dim=-1)
        comb = self.comb_proj(comb)

        output = self.self_attn(comb, position)

        return output
    

