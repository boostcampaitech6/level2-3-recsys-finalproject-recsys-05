import os
import random
import numpy as np
import torch



class CFG:
    def __init__(self, config_file: str = None):
        self._store = {}
        if config_file is not None:
            extension = config_file.split(sep='.')[-1]
            if extension == 'json':
                import json
                form = json
                with open(config_file, 'r') as f:
                    config = form.load(f)

            elif extension == 'yaml':
                import yaml
                form = yaml
                with open(config_file, 'r') as f:
                    config = form.load(f, Loader=yaml.FullLoader)

            else:
                raise TypeError

            for key, value in config.items():
                self._store[key] = value
    

    def __getitem__(self, key):
        return self._store[key]
    

    def __setitem__(self, key, value):
        self._store[key] = value


    def __repr__(self):
        return str(self._store)
    

def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", type=bool, help="Use GPU")
    parser.add_argument("--data_dir", type=str, help="")
    parser.add_argument("--output_dir", type=str, help="")
    parser.add_argument("--model_dir", type=str, help="")
    parser.add_argument("--model_name", type=str, help="")
    
    parser.add_argument("--batch_size", type=int, help="")
    parser.add_argument("--emb_size", type=int, help="")
    parser.add_argument("--hidden_size", type=int, help="")
    parser.add_argument("--n_layers", type=int, help="")
    parser.add_argument("--n_head", type=int, help="")
    parser.add_argument("--seq_len", type=int, help="")
    parser.add_argument("--n_epochs", type=int, help="")
    parser.add_argument("--lr", type=float, help="")
    parser.add_argument("--dropout", type=float, help="")

    parser.add_argument("--verbose", type=bool, help="")
    

    args = parser.parse_args()

    return args


def set_seeds(seed: int = 0):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_for_distributed(opts):
    import torch.distributed as dist
    
    # 1. setting for distributed training
    opts.global_rank = int(os.environ['RANK'])
    opts.local_rank = int(os.environ['LOCAL_RANK'])
    opts.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(opts.local_rank)
    if opts.global_rank is not None and opts.local_rank is not None:
        print("Use GPU: [{}/{}] for training".format(opts.global_rank, opts.local_rank))

    # 2. init_process_group
    dist.init_process_group(backend="nccl")


def get_dataloader(cfg, dataset) -> tuple:
    from SE.dataset import custom_collate_fn
    from torch.utils.data import DataLoader, Subset
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * cfg['train_ratio'])
    valid_size = dataset_size - train_size

    indices = torch.randperm(dataset_size).tolist()

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=custom_collate_fn)

    return train_loader, valid_loader
