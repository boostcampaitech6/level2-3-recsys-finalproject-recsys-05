import pandas as pd
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from src.util import CFG, parse_args, init_for_distributed
from src.dataset import SASDataset
from src.model import SASModel
from src.train import run, get_dataloader
from src.loss import CosLoss
from src.util import get_logger, logging_conf
import wandb

logger = get_logger(logging_conf)

def main(cfg: CFG):
    if dist.get_rank() == 0:
        wandb.init(project=cfg['wandb_project_name'])

    df = pd.read_csv(os.path.join(cfg['data_dir'], f'riot_match.csv.gzip'), compression='gzip')
    logger.info("## csv data loaded")

    cfg['n_layers'] = df[cfg['cate_cols']].max().max() + 1

    dataset = SASDataset(cfg, df)
    logger.info("## dataset loaded ")

    train_loader, valid_loader = get_dataloader(cfg, dataset)

    model = SASModel(cfg).cuda(cfg['local_rank'])
    model = DDP(module=model,
                device_ids=[cfg['local_rank']])
                
    logger.info("## model loaded ")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['lr'])
    loss_fun = CosLoss()

    run(model, train_loader, valid_loader, optimizer, loss_fun, cfg)

if __name__ == '__main__':
    torch.cuda.empty_cache()

    args = parse_args()
    cfg = CFG('config.yaml')

    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    init_for_distributed(cfg)
    main(cfg)