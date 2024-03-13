import pandas as pd
import torch
from SE.util import CFG, parse_args, init_for_distributed, get_dataloader
from SE.dataset import SimilarityDataset


def main(cfg: CFG):
    tier = 'diamond'
    match_df = pd.read_csv(f'../data/{tier}_match_by_match_mod.csv')
    summoner_df = pd.read_csv(f'../data/{tier}_match_by_summoner_mod.csv')

    dataset = SimilarityDataset(cfg, summoner_df, match_df)
    train_loader, valid_loader = get_dataloader(cfg, dataset)

if __name__ == '__main__':
    args = parse_args()
    cfg = CFG()

    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    init_for_distributed(cfg)
    main(cfg)