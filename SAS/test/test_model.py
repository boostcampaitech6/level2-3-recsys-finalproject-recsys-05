import random
import unittest
import pandas as pd
import os
import torch
from src.dataset import SASDataset
from torch.utils.data import DataLoader
from src.model import SASModel
from src.loss import STD_loss
from src.util import CFG


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.cfg = CFG('test/config.yaml')
        self.device = self.cfg['device']

        random.seed(0)

        tier = 'diamond'
        self.match_df = pd.read_csv(os.path.join(self.cfg['data_dir'], f'{tier}_match_by_match_mod.csv'), compression='gzip')
        self.summoner_df = pd.read_csv(os.path.join(self.cfg['data_dir'], f'{tier}_match_by_summoner_mod.csv'), compression='gzip')
        self.cfg['n_layers'] = self.match_df[self.cfg['cate_cols']].max().max() + 1


    def test_model(self):
        dataset = SASDataset(self.cfg, self.summoner_df, self.match_df)
        data_loader = DataLoader(dataset=dataset, batch_size=3)
        model = SASModel(self.cfg).to(self.device)
        loss_fun = STD_loss()

        model.train()
        torch.set_grad_enabled(True)

        for cate, cont, pos in data_loader:
            cate, cont, pos = cate.to(self.device), cont.to(self.device), pos.to(self.device)

            output = model(cate, cont, pos)

            loss = loss_fun(output)
            # print(output)
            print(loss.item())
            
        assert()
        


if __name__ == '__main__':
    unittest.main()