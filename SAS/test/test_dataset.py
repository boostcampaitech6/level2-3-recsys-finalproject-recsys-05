import random
import unittest
import pandas as pd
import os
import torch
from src.dataset import SASDataset
from src.util import CFG, df_shift_index
import pymongo
from src.MongoDB_Controller import MongoDBController


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.cfg = CFG('test/config.yaml')
        
        client = pymongo.MongoClient("mongodb://teemo:ui6287@power16one5.iptime.org:27017/loldb")
        batch_size = 100000
        mongo = MongoDBController(client, batch_size)

        self.df = pd.read_csv(os.path.join(self.cfg['data_dir'], f'riot_match.csv.gzip'), compression='gzip')

        self.cfg['n_layers'] = self.df[self.cfg['cate_cols']].max().max() + 1

        
    def test_dataset(self):
        dataset = SASDataset(self.cfg, self.df, self.cfg['cate_cols'])

        index = random.randint(0, len(dataset))

        cate, cont, pos = dataset[index]
  
        # print(cate.shape)
        # print(cate[:2, :, (0, 1, 2)])
        
        # summoner_pos = pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, cate.size(2))
        # target_summoner = torch.gather(cate, 1, summoner_pos)
        # print(target_summoner[:, :, (0, 1, 2)])

        # self.assertEqual(cate[0, 0, 0])
        # print(cont[0, 0])
        # print(pos)
        assert()
        


if __name__ == '__main__':
    unittest.main()