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


def get_loss(model: nn.Module, data_loader: DataLoader, loss_fun: nn.Module, device: str, is_train: bool, optimizer: torch.optim.Optimizer = None):
    if is_train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    total_loss = 0
    target_list = []
    output_list = []
    for A_cate, A_cont, B_cate, B_cont, result in tqdm(data_loader):
        A_cate, A_cont, B_cate, B_cont, result = A_cate.to(device), A_cont.to(device), B_cate.to(device), B_cont.to(device), result.to(device)

        if is_train:
            optimizer.zero_grad()
        output1 = model(A_cate, A_cont)
        output2 = model(B_cate, B_cont)

        output = F.cosine_similarity(output1, output2, dim=1)
        output = torch.abs(output)
        
        loss = loss_fun(output, result) 
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        target_list.append(result.detach().cpu())
        output_list.append(output.detach().cpu())

    target_list = torch.concat(target_list).numpy()
    output_list = torch.concat(output_list).numpy()
    
    acc = accuracy_score(y_true=target_list, y_pred=output_list > 0.5)
    auc = roc_auc_score(y_true=target_list, y_score=output_list)
    average_loss = total_loss / len(data_loader)

    avg_total_acc, avg_total_auc, avg_total_loss = evaluate(acc, auc, average_loss, device)

    if is_train:
        mode = "TRAIN"
    else:
        mode = "VALID"
    logger.info("%s AUC : %.4f, ACC : %.4f, LOSS : %.4f", mode, avg_total_acc, avg_total_auc, avg_total_loss)

    return avg_total_acc, avg_total_auc, avg_total_loss


def save_model(model, cfg, epoch, metric):
    os.makedirs(cfg['model_dir'], exist_ok=True)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    torch.save(obj= {"model": model.state_dict(), "epoch": epoch + 1},
            f=os.path.join(cfg['model_dir'], f"{metric:.2%}_{now}.pt"))


def run(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fun: nn.Module, cfg):
    logger.info(f"Training Started : n_epochs={cfg['n_epochs']}")
    best_auc, best_acc, best_epoch = 0, 0, -1
    for epoch in range(cfg['n_epochs']):
        logger.info("Epoch: %s", epoch)
        # TRAIN
        train_auc, train_acc, train_loss = get_loss(data_loader=train_loader, model=model, optimizer=optimizer, loss_fun=loss_fun, device = cfg['device'], is_train=True)
    
        # VALID
        valid_auc, valid_acc, valid_loss = get_loss(model=model, data_loader=valid_loader, loss_fun=loss_fun, device = cfg['device'], is_train=False)

        if dist.get_rank() == 0:
            wandb.log(dict(train_acc_epoch=train_acc,
                            train_auc_epoch=train_auc,
                            train_loss_epoch=train_loss,
                            valid_acc_epoch=valid_acc,
                            valid_auc_epoch=valid_auc,
                            valid_loss_epoch=valid_loss))

        if valid_auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, valid_auc)
            best_auc, best_epoch = valid_auc, epoch

            if dist.get_rank() == 0:
                save_model(model, cfg, epoch, valid_auc)
            
            ### early stopping
            cur_step = 0
        elif valid_acc > best_acc:
            logger.info("Best model updated ACC from %.4f to %.4f", best_acc, valid_acc)
            best_acc, best_epoch = valid_acc, epoch

            if dist.get_rank() == 0:
                save_model(model, cfg, epoch, valid_acc)
            
            ### early stopping
            cur_step = 0
        else:
            cur_step += 1
            if cur_step > cfg['max_steps']:
                logger.info(f"Early Stopping at {epoch+1}'th epoch")
                break
               
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")
