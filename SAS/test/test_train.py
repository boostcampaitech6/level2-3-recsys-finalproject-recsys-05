import random
import unittest
import pandas as pd
import os
import torch
from src.dataset import SASDataset
from torch.utils.data import DataLoader
from src.model import SASModel
from src.loss import CosLoss
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


    def test_train(self):
        dataset = SASDataset(self.cfg, self.summoner_df, self.match_df)
        data_loader = DataLoader(dataset=dataset, batch_size=3)
        model = SASModel(self.cfg).to(self.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.cfg['lr'])
        loss_fun = CosLoss()

        model.train()
        torch.set_grad_enabled(True)

        prev_output = None
        for cate, cont, pos in data_loader:
            cate, cont, pos = cate.to(self.device), cont.to(self.device), pos.to(self.device)

            optimizer.zero_grad()

            output = model(cate, cont, pos)

            if prev_output is None:
                prev_output = output.detach()
                continue

            loss = loss_fun(output, prev_output)

            loss.backward()
            optimizer.step()

            prev_output = output.detach()

            # print(output)
            print(loss.item())
            
        assert()
        


if __name__ == '__main__':
    unittest.main()