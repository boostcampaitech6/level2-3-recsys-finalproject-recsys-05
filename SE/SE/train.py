import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from SE.util import get_logger, logging_conf
from tqdm import tqdm
import os
from datetime import datetime
import wandb


logger = get_logger(logging_conf)


def get_dataloader(cfg, dataset) -> tuple:
    from SE.dataset import custom_collate_fn
    from torch.utils.data import DistributedSampler
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * cfg['train_ratio'])
    valid_size = dataset_size - train_size

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
                            #   num_workers=int(cfg['num_workers'] / cfg['world_size']),
                              sampler=train_sampler,
                              collate_fn=custom_collate_fn,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=cfg['batch_size'], 
                              num_workers=cfg['num_workers'],
                            #   num_workers=int(cfg['num_workers'] / cfg['world_size']),
                              sampler=valid_sampler,
                              collate_fn=custom_collate_fn,
                              pin_memory=True)

    return train_loader, valid_loader


def evaluate(acc, auc, average_loss, device):
    total_acc = torch.tensor(acc).to(device)
    total_auc = torch.tensor(auc).to(device)
    total_loss = torch.tensor(average_loss).to(device)

    dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_auc, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    avg_total_acc = total_acc / world_size
    avg_total_auc = total_auc / world_size
    avg_total_loss = total_loss / world_size

    return avg_total_acc, avg_total_auc, avg_total_loss


def train(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fun: nn.Module, device: str):
    model.train()
    torch.set_grad_enabled(True)

    total_loss = 0
    target_list = []
    output_list = []
    for A_cate, A_cont, B_cate, B_cont, result in tqdm(train_loader):
        A_cate, A_cont, B_cate, B_cont, result = A_cate.to(device), A_cont.to(device), B_cate.to(device), B_cont.to(device), result.to(device)

        optimizer.zero_grad()
        output1 = model(A_cate, A_cont)
        output2 = model(B_cate, B_cont)

        output = F.cosine_similarity(output1, output2, dim=1)
        output = torch.abs(output)
        
        loss = loss_fun(output, result) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        target_list.append(result.detach().cpu())
        output_list.append(output.detach().cpu())

    target_list = torch.concat(target_list).numpy()
    output_list = torch.concat(output_list).numpy()
    
    acc = accuracy_score(y_true=target_list, y_pred=output_list > 0.5)
    auc = roc_auc_score(y_true=target_list, y_score=output_list)
    average_loss = total_loss / len(train_loader)

    avg_total_acc, avg_total_auc, avg_total_loss = evaluate(acc, auc, average_loss, device)

    logger.info("TRAIN AUC : %.4f, ACC : %.4f, LOSS : %.4f", avg_total_acc, avg_total_auc, avg_total_loss)

    return avg_total_acc, avg_total_auc, avg_total_loss


def validate(model: nn.Module, valid_loader: DataLoader, loss_fun: nn.Module, device: str):
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0
    target_list = []
    output_list = []
    for A_cate, A_cont, B_cate, B_cont, result in tqdm(valid_loader):
        A_cate, A_cont, B_cate, B_cont, result = A_cate.to(device), A_cont.to(device), B_cate.to(device), B_cont.to(device), result.to(device)
        output1 = model(A_cate, A_cont)
        output2 = model(B_cate, B_cont)

        output = F.cosine_similarity(output1, output2, dim=1)
        output = torch.abs(output)

        loss = loss_fun(output, result)
        total_loss += loss.item()
        target_list.append(result.detach().cpu())
        output_list.append(output.detach().cpu())

    target_list = torch.concat(target_list).numpy()
    output_list = torch.concat(output_list).numpy()

    acc = accuracy_score(y_true=target_list, y_pred=output_list > 0.5)
    auc = roc_auc_score(y_true=target_list, y_score=output_list)
    average_loss = total_loss / len(valid_loader)

    avg_total_acc, avg_total_auc, avg_total_loss = evaluate(acc, auc, average_loss, device)

    logger.info("VALID AUC : %.4f, ACC : %.4f, LOSS : %.4f", avg_total_acc, avg_total_auc, avg_total_loss)

    return avg_total_acc, avg_total_auc, avg_total_loss


def run(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fun: nn.Module, cfg):
    logger.info(f"Training Started : n_epochs={cfg['n_epochs']}")
    best_acc, best_epoch = 0, -1
    for e in range(cfg['n_epochs']):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader, model=model, optimizer=optimizer, loss_fun=loss_fun, device = cfg['device'])
    
        # VALID
        valid_auc, valid_acc, valid_loss = validate(model=model, valid_loader=valid_loader, loss_fun=loss_fun, device = cfg['device'])

        if dist.get_rank() == 0:
            wandb.log(dict(train_acc_epoch=train_acc,
                            train_auc_epoch=train_auc,
                            train_loss_epoch=train_loss,
                            valid_acc_epoch=valid_acc,
                            valid_auc_epoch=valid_auc,
                            valid_loss_epoch=valid_loss))

        if valid_acc > best_acc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_acc, valid_acc)
            best_acc, best_epoch = valid_acc, e

            if dist.get_rank() == 0:
                os.makedirs(cfg['model_dir'], exist_ok=True)
                now = datetime.now().strftime("%Y_%m_%d_%H_%M")
                torch.save(obj= {"model": model.state_dict(), "epoch": e + 1},
                        f=os.path.join(cfg['model_dir'], f"{valid_acc:.2%}_{now}.pt"))
            
            ### early stopping
            cur_step = 0
        else:
            cur_step += 1
            if cur_step > cfg['max_steps']:
                logger.info(f"Early Stopping at {e+1}'th epoch")
                break
               
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")