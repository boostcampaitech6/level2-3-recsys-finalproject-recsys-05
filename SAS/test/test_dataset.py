import random
import unittest
import pandas as pd
import os
import torch
from src.dataset import SASDataset
from torch.utils.data import DataLoader
from src.util import CFG


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.cfg = CFG('test/config.yaml')

        self.df = pd.read_csv(os.path.join(self.cfg['data_dir'], f'SAS_test_data_v1.csv'))

        layer_col = [x for x in self.cfg['cate_cols'] if x != 'summonerId' and x != 'matchId' and x != 'position']
        self.cfg['n_layers'] = self.df[layer_col].max().max() + 1

        
    def test_dataset(self):
        dataset = SASDataset(self.cfg, self.df, ['summonerId', 'matchId', 'position'])
        data_loader = DataLoader(dataset=dataset, batch_size=10)

        index = random.randint(0, len(dataset))

        cate, cont, pos = dataset[index]
  
        summoner_pos = pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, cate.size(2))
        target_summoner = torch.gather(cate, 1, summoner_pos)
        target_summoner = target_summoner.squeeze()[:, 0]

        target_summoner = [int(target_summoner[i]) for i in range(len(target_summoner))]
        target_summoner = [target_summoner[0] == x for x in target_summoner[1:]]
        
        self.assertTrue(all([target_summoner]))

        for cate, cont, pos in data_loader:
            break


if __name__ == '__main__':
    unittest.main()