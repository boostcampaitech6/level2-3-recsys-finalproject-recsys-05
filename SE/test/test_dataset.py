import unittest
import pandas as pd
from SE.dataset import SimilarityDataset
from SE.util import CFG


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.cfg = CFG('test/test_config.yaml')

        tier = 'diamond'
        self.match_df = pd.read_csv(f'../data/{tier}_match_by_match_mod.csv')
        self.summoner_df = pd.read_csv(f'../data/{tier}_match_by_summoner_mod.csv')

    def test_GTdataset(self):
        dataset = SimilarityDataset(self.cfg, self.summoner_df, self.match_df)

        num = 1
        print(dataset[num])
        dataset
        assert()



if __name__ == '__main__':
    unittest.main()