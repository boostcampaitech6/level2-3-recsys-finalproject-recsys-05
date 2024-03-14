import torch
from torch import nn

class SimilarityModel(nn.Module):
    def __init__(self, cfg):
        super(SimilarityModel, self).__init__()
        
        # categorical
        self.cate_emb = nn.Embedding(cfg['n_layers'], cfg['emb_size'], padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg['max_seq_len'] * cfg['emb_size'] * len(cfg['cate_col']), cfg['hidden_size']),
            nn.LayerNorm(cfg['hidden_size']),
            nn.Dropout(p=cfg['dropout'])
        )

        # continuous
        self.cont_proj = nn.Sequential(
            nn.BatchNorm1d(cfg['max_seq_len'] * len(cfg['cont_col'])),
            nn.Linear(cfg['max_seq_len'] * len(cfg['cont_col']), cfg['hidden_size']),
            nn.LayerNorm(cfg['hidden_size']),
            nn.Dropout(p=cfg['dropout'])
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg['hidden_size'] * 2, cfg['hidden_size']),
            nn.LayerNorm(cfg['hidden_size']),
            nn.Dropout(p=cfg['dropout'])
        )


    def forward(self, cate, cont):
        size = [i for i in cate.size()]
        cate = self.cate_emb(cate).view(*size[:-1], -1)
        cate = self.cate_proj(cate)
        
        cont = self.cont_proj(cont)

        output = self.comb_proj(torch.cat([cate, cont], dim=1))

        return output