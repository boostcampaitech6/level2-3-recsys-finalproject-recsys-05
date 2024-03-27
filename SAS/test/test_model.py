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

        self.df = pd.read_csv(os.path.join(self.cfg['data_dir'], self.cfg['data_file']))

        layer_col = [x for x in self.cfg['cate_cols'] if x != 'summonerId' and x != 'matchId' and x != 'position']
        self.cfg['n_layers'] = self.df[layer_col].max().max() + 1


    def test_model(self):
        dataset = SASDataset(self.cfg, self.df)
        data_loader = DataLoader(dataset=dataset, batch_size=3)
        model = SASModel(self.cfg).to(self.device)
        loss_fun = CosLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.cfg['lr'])

        model.train()
        torch.set_grad_enabled(True)

        is_train = True
        prev_output = None
        total_loss = 0
        for cate, cont, pos in data_loader:
            cate, cont, pos = cate.to(self.device), cont.to(self.device), pos.to(self.device)

            if is_train:
                optimizer.zero_grad()

            output = model(cate, cont, pos)
            
            if prev_output is None:
                prev_output = output.detach()
                continue
        
            if output.size(0) != prev_output.size(0):
                # print(f"Skipping loss calculation due to size mismatch.")
                prev_output = output.detach()
                continue

            loss = loss_fun(output, prev_output) 

            if is_train:
                loss.backward()
                optimizer.step()

            prev_output = output.detach()

            total_loss += loss.item()

            print(f"Loss: {loss.item()}")

        assert()
        


if __name__ == '__main__':
    unittest.main()