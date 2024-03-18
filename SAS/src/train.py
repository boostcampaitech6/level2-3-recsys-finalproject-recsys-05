import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from src.util import get_logger, logging_conf
from tqdm import tqdm
import os
from datetime import datetime
import wandb


logger = get_logger(logging_conf)


def get_dataloader(cfg, dataset) -> tuple:
    from torch.utils.data import DistributedSampler
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * cfg['train_ratio'])

    indices = torch.randperm(dataset_size).tolist()

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=True)

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=cfg['batch_size'], 
                              num_workers=cfg['num_workers'],
                              sampler=train_sampler,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=cfg['batch_size'], 
                              num_workers=cfg['num_workers'],
                              sampler=valid_sampler,
                              pin_memory=True)

    return train_loader, valid_loader


def evaluate(average_loss, device):
    total_loss = torch.tensor(average_loss).to(device)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    avg_total_loss = total_loss / world_size

    return avg_total_loss


def get_loss(model: nn.Module, data_loader: DataLoader, loss_fun: nn.Module, device: str, is_train: bool, optimizer: torch.optim.Optimizer = None):
    if is_train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    total_loss = 0
    for cate, cont, pos in tqdm(data_loader):
        cate, cont, pos = cate.to(device), cont.to(device), pos.to(device)

        if is_train:
            optimizer.zero_grad()
        
        output = model(cate, cont, pos)
        
        loss = loss_fun(output) 

        if is_train:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        total_loss += loss.item()
    
    average_loss = total_loss / len(data_loader)

    avg_total_loss = evaluate(average_loss, device)

    if is_train:
        mode = "TRAIN"
    else:
        mode = "VALID"
    logger.info("%s LOSS : %.4f", mode, avg_total_loss)

    return avg_total_loss


def save_model(model, cfg, epoch, metric):
    os.makedirs(cfg['model_dir'], exist_ok=True)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    torch.save(obj= {"model": model.state_dict(), "epoch": epoch + 1},
            f=os.path.join(cfg['model_dir'], f"{metric:.2%}_{now}.pt"))


def run(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fun: nn.Module, cfg):
    logger.info(f"Training Started : n_epochs={cfg['n_epochs']}")
    best_loss, best_epoch, cur_step = 99999, -1, 0
    for epoch in range(cfg['n_epochs']):
        logger.info("Epoch: %s", epoch)
        # TRAIN
        train_loss = get_loss(data_loader=train_loader, model=model, optimizer=optimizer, loss_fun=loss_fun, device = cfg['device'], is_train=True)
    
        # VALID
        valid_loss = get_loss(model=model, data_loader=valid_loader, loss_fun=loss_fun, device = cfg['device'], is_train=False)

        if dist.get_rank() == 0:
            wandb.log(dict(train_loss_epoch=train_loss,
                            valid_loss_epoch=valid_loss))

        if valid_loss > best_loss:
            logger.info("Best model updated LOSS from %.4f to %.4f", best_loss, valid_loss)
            best_loss, best_epoch = valid_loss, epoch

            if dist.get_rank() == 0:
                save_model(model, cfg, epoch, valid_loss)
            
            ### early stopping
            cur_step = 0
        else:
            cur_step += 1
            if cur_step > cfg['max_steps']:
                logger.info(f"Early Stopping at {epoch+1}'th epoch")
                break
               
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")
