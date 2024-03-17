import random
import unittest
import pandas as pd
import os
from src.dataset import SASDataset
from src.util import CFG


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.cfg = CFG('test/config.yaml')

        # random.seed(0)

        tier = 'diamond'
        self.match_df = pd.read_csv(os.path.join(self.cfg['data_dir'], f'{tier}_match_by_match_mod.csv'), compression='gzip')
        self.summoner_df = pd.read_csv(os.path.join(self.cfg['data_dir'], f'{tier}_match_by_summoner_mod.csv'), compression='gzip')

    def test_dataset(self):
        dataset = SASDataset(self.cfg, self.summoner_df, self.match_df)

        index = random.randint(0, len(dataset))

        cate, cont, pos = dataset[index]
  
        print(pos)
        assert()
        


if __name__ == '__main__':
    unittest.main()