import pandas as pd
from sympy import comp
import torch
import torch.nn as nn
from torch import distributed as dist
import os
from SE.util import CFG, parse_args, init_for_distributed
from SE.dataset import SimilarityDataset
from SE.model import SimilarityModel
from SE.train import run, get_dataloader
import wandb


def main(cfg: CFG):
    if dist.get_rank() == 0:
        wandb.init(project=cfg['wandb_project_name'])

    tier = cfg['tier']
    match_df = pd.read_csv(os.path.join(cfg['data_dir'], tier, f'match_by_match.csv'), compression='gzip')
    summoner_df = pd.read_csv(os.path.join(cfg['data_dir'], tier, f'match_by_summoner.csv'), compression='gzip')

    cfg['n_layers'] = summoner_df[cfg['cate_cols']].max().max() + 1


    dataset = SimilarityDataset(cfg, summoner_df, match_df)
    train_loader, valid_loader = get_dataloader(cfg, dataset)

    model = SimilarityModel(cfg).cuda(cfg['local_rank'])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['lr'])
    loss_fun = nn.CrossEntropyLoss()

    run(model, train_loader, valid_loader, optimizer, loss_fun, cfg)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    cfg = CFG('config.yaml')

    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    init_for_distributed(cfg)
    main(cfg)