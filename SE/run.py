import pandas as pd
from sympy import comp
import torch
import torch.nn as nn
from SE.util import CFG, parse_args, init_for_distributed
from SE.dataset import SimilarityDataset
from SE.model import SimilarityModel
from SE.train import run, get_dataloader


def main(cfg: CFG):
    tier = 'diamond'
    match_df = pd.read_csv(f'../data/{tier}_match_by_match_mod.csv', compression='gzip')
    summoner_df = pd.read_csv(f'../data/{tier}_match_by_summoner_mod.csv', compression='gzip')

    dataset = SimilarityDataset(cfg, summoner_df, match_df)
    train_loader, valid_loader = get_dataloader(cfg, dataset)

    model = SimilarityModel(cfg)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['lr'])
    loss_fun = nn.CrossEntropyLoss()

    run(model, train_loader, valid_loader, optimizer, loss_fun, n_epochs=100, model_dir="model", max_step=5)


if __name__ == '__main__':
    args = parse_args()
    cfg = CFG('config.yaml')

    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    init_for_distributed(cfg)
    main(cfg)