import torch
import torch.nn as nn
import torch.nn.functional as F
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
                              shuffle=True,
                              num_workers=cfg['num_workers'],
                              sampler=train_sampler,
                              collate_fn=custom_collate_fn,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=cfg['batch_size'], 
                              shuffle=True,
                              num_workers=cfg['num_workers'],
                              sampler=valid_sampler,
                              collate_fn=custom_collate_fn,
                              pin_memory=True)

    return train_loader, valid_loader


def train(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fun: nn.Module):
    model.train()
    torch.set_grad_enabled(True)

    total_loss = 0
    target_list = []
    output_list = []
    for A_cate, A_cont, B_cate, B_cont, result in tqdm(train_loader):
        optimizer.zero_grad()
        output1 = model(A_cate, A_cont)
        output2 = model(B_cate, B_cont)

        output = F.cosine_similarity(output1, output2, dim=1)
        
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

    logger.info("TRAIN AUC : %.4f, ACC : %.4f, LOSS : %.4f", auc, acc, average_loss)

    return auc, acc, average_loss


def validate(model: nn.Module, valid_loader: DataLoader, loss_fun: nn.Module):
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0
    target_list = []
    output_list = []
    for A_cate, A_cont, B_cate, B_cont, result in tqdm(valid_loader):
        output1 = model(A_cate, A_cont)
        output2 = model(B_cate, B_cont)

        output = F.cosine_similarity(output1, output2, dim=1)

        loss = loss_fun(output, result)
        total_loss += loss.item()
        target_list.append(result.detach().cpu())
        output_list.append(output.detach().cpu())

    target_list = torch.concat(target_list).numpy()
    output_list = torch.concat(output_list).numpy()

    acc = accuracy_score(y_true=target_list, y_pred=output_list > 0.5)
    auc = roc_auc_score(y_true=target_list, y_score=output_list)
    average_loss = total_loss / len(valid_loader)

    logger.info("VALID AUC : %.4f, ACC : %.4f, LOSS : %.4f", auc, acc, average_loss)

    return auc, acc, average_loss


def run(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fun: nn.Module, n_epochs: int = 100, model_dir: str = "model", max_step: int = 5):
    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader, model=model, optimizer=optimizer, loss_fun=loss_fun)
    
        # VALID
        valid_auc, valid_acc, valid_loss = validate(model=model, valid_loader=valid_loader, loss_fun=loss_fun)

        wandb.log(dict(train_acc_epoch=train_acc,
                       train_auc_epoch=train_auc,
                       train_loss_epoch=train_loss,
                       valid_acc_epoch=valid_auc,
                       valid_auc_epoch=valid_acc,
                       valid_loss_epoch=valid_loss))

        if valid_auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, valid_auc)
            best_auc, best_epoch = valid_auc, e
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1},
                       f=os.path.join(model_dir, f"{valid_auc:.2%}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.pt"))
            
            ### early stopping
            cur_step = 0
        else:
            cur_step += 1
            if cur_step > max_step:
                logger.info(f"Early Stopping at {e+1}'th epoch")
                break
               
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")
