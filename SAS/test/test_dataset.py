import random
import unittest
import pandas as pd
from src.dataset import SASDataset
from src.util import CFG


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.cfg = CFG('test/config.yaml')

        random.seed(0)

        tier = 'diamond'
        self.match_df = pd.read_csv(f'~/data2/test/{tier}_match_by_match_mod.csv', compression='gzip')
        self.match_df = self.match_df.sort_values(by=['match_id', 'team_key', 'position']).reset_index(drop=True)
        self.summoner_df = pd.read_csv(f'~/data2/test/{tier}_match_by_summoner_mod.csv', compression='gzip')
        self.summoner_df = self.summoner_df.sort_values(by=['summoner_id']).reset_index(drop=True)

    def test_GTdataset(self):
        dataset = SASDataset(self.cfg, self.summoner_df, self.match_df)

        index = random.randint(0, len(dataset))

        cate, cont, pos = dataset[index]
  
        print(cate, pos)
        assert()
        


if __name__ == '__main__':
    unittest.main()