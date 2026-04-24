import os
import torch
import logging
import datetime
import numpy as np
import pandas as pd

from pathlib import Path
from sconf import Config
from transformers.utils import ModelOutput
from torcheval.metrics import MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from typing import Any, List, Tuple, Optional, Union

from pytorch_lightning.utilities import rank_zero_only
#--------------------------------------------------#

'''
--- [def] list ---
save_config_file(config: dict, path: Union[str, os.PathLike], time: str, verbose: Optional[bool]=True):
save_model_results(config, path, model, epoch, val_loss, best_loss, result, verbose, name:str, codebook=None):
save_recon_results(path, target, recon, epoch, name):
recon_results(target, recon, target_sent: torch.Tensor, recon_sent: list):
get_loss(result:dict, val_result:dict):
get_train_loss(result:dict) -> dict:
get_val_loss(val_result:dict) -> dict:
get_reg_loss(result:dict, val_result:dict, ext_acc=None, val_ext_acc=None):
get_rec_loss(result:dict, val_result:dict):
print_info(losses: dict):
print_rec_info(losses: dict):
print_reg_info(losses: dict):
log(writer:SummaryWriter, tag:str, scalar_value:list, global_step=None, walltime=None, new_style=False, double_precision=False):
get_time_duration(start, end):
get_requires_grad(path, model):
get_mae(logits: np.array, target:np.array):
get_onehot(pred):
sigmoid(x):
get_acc(logits: np.array, target:np.array) -> np.array:
get_accuracy(pred:list, target:list) -> list:
-----------------
'''

@rank_zero_only
def save_config_file(config: dict, path: Union[str, os.PathLike], time: str, verbose: Optional[bool]=True):
    """saving config file with sconf Config

    Args:
        config (Config): configs to be saved
        path (Union[str, os.PathLike]): path to save config.yaml file (/saved)
        verbose (Optional[bool], optional): dump the config file or not (Default: `True`)
    """
    if not Path(path).exists():
        os.makedirs(path)
    SAVE_PATH = Path(path) / f'config_{time}.yaml'
    
    config = Config(config.__dict__)
    
    # whether to print at terminal
    if verbose:
        print("---------------------------")
        print(config.dumps()) # save config into json file format
        print("---------------------------")
    
    with open(SAVE_PATH, 'w') as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"[SAVED] Config is saved at ({SAVE_PATH})")
        print()


@rank_zero_only
def save_model_results(config, path, model, epoch, val_loss, best_loss:float, name:str, codebook=None):
    
    """saving model pth file and train/val results

    Args:
        path: path for files to be saved
        model: model to be saved
        epoch: current epoch
        val_loss: current validation loss value
        best_loss: previous lowest loss value
        result (dict): result information dictionary
        time (str): time used in file name
        verbose (bool): whether to save the result or not
        name (str): name of the saved weight file
        codebook (numpy)
    """
    print(f'[BEST VALID LOSS UPDATED | EPOCH {epoch+1}]: {best_loss:.4f} -> {val_loss:.4f}')
    
    # define saved path
    if not Path(path).exists():
        os.makedirs(path)
    SAVE_MODEL_PATH = Path(path)
    
    # save model
    if not config.debug:
        torch.save(model.state_dict(), SAVE_MODEL_PATH / f'{name}.pth')
        print(f'[MODEL SAVED TO] {SAVE_MODEL_PATH} IN EPOCH {epoch+1}')
    
    # save codebook (stop saving 240717)
    # if codebook is not None:
    #     SAVE_Q_PATH = Path(path) / 'codebook'
    #     if not Path(SAVE_Q_PATH).exists():
    #         os.makedirs(SAVE_Q_PATH)
        
    #     np.save(SAVE_Q_PATH / f'{time}_codebook_{epoch+1}.npy', codebook.clone().detach().cpu())


def save_recon_results(path, target, recon, epoch, name):
    SAVE_SENT_PATH = Path(path) / 'sentence'
    if not Path(SAVE_SENT_PATH).exists():
        os.makedirs(SAVE_SENT_PATH)
    
    np.save(SAVE_SENT_PATH / f'{name}_target_e{epoch}.npy', target)
    np.save(SAVE_SENT_PATH / f'{name}_recon_e{epoch}.npy', recon)
    
    print(f"[SAVED] Sentences are saved at ({SAVE_SENT_PATH})")


def recon_results(target, recon, target_sent: torch.Tensor, recon_sent: list):
    
    target_sent = target_sent.cpu().detach().tolist()
    recon_sent = recon_sent.cpu().detach().tolist()
    
    # add new sentences
    target = target + target_sent
    recon = recon + recon_sent

    return target, recon


def get_loss(result:dict, val_result:dict):
    """getting mean value of train/val result

    Args:
        result (dict): train results
        val_result (dict): validation results

    Returns:
        mean value of each loss
    """
    losses = {
                "r_loss": 0,
                "val_r_loss": 0,
                "r_recon_loss": 0,
                "val_r_recon_loss": 0,
                "r_emb_loss": 0,
                "val_r_emb_loss": 0,
                "r_label_loss": 0,
                "val_r_label_loss": 0,
                "r_label_loss_2": 0,
                "val_r_label_loss_2": 0,
                "ppl": 0,
                "val_ppl":0
            }
    
    losses["r_loss"] = sum(result["loss"])/len(result["loss"])
    losses["val_r_loss"] = sum(val_result["loss"])/len(val_result["loss"])
    
    losses["r_recon_loss"] = sum(result["recon_loss"])/len(result["recon_loss"])
    losses["val_r_recon_loss"] = sum(val_result["recon_loss"])/len(val_result["recon_loss"])
    
    losses["r_emb_loss"] = sum(result["emb_loss"])/len(result["emb_loss"])
    losses["val_r_emb_loss"] = sum(val_result["emb_loss"])/len(val_result["emb_loss"])
    
    losses["r_label_loss"] = sum(result["label_loss"])/len(result["label_loss"])
    losses["val_r_label_loss"] = sum(val_result["label_loss"])/len(val_result["label_loss"])
    
    losses["r_label_loss_2"] = sum(result["label_loss_2"])/len(result["label_loss_2"])
    losses["val_r_label_loss_2"] = sum(val_result["label_loss_2"])/len(val_result["label_loss_2"])
    
    losses["ppl"] = sum(result["perplexity"])/len(result["perplexity"])
    losses["val_ppl"] = sum(val_result["perplexity"])/len(val_result["perplexity"])
    
    return losses

def get_train_loss(result:dict) -> dict:
    """getting mean value of the train result

    Args:
        result (dict): train results

    Returns:
        mean value of each loss
    """
    losses = {
                "r_loss": 0,
                "r_recon_loss": 0,
                "r_emb_loss": 0,
                "r_label_loss": 0,
                "r_label_loss_2": 0,
                "ppl": 0,
            }
    
    losses["r_loss"] = sum(result["loss"])/len(result["loss"])
    
    losses["r_recon_loss"] = sum(result["recon_loss"])/len(result["recon_loss"])
    
    losses["r_emb_loss"] = sum(result["emb_loss"])/len(result["emb_loss"])
    
    losses["r_label_loss"] = sum(result["label_loss"])/len(result["label_loss"])
    
    losses["r_label_loss_2"] = sum(result["label_loss_2"])/len(result["label_loss_2"])
    
    losses["ppl"] = sum(result["perplexity"])/len(result["perplexity"])
    
    return losses 

def get_val_loss(val_result:dict) -> dict:
    """getting mean value of the validation result

    Args:
        val_result (dict): validation results

    Returns:
        mean value of each loss
    """
    losses = {
                "val_r_loss": 0,
                "val_r_recon_loss": 0,
                "val_r_emb_loss": 0,
                "val_r_label_loss": 0,
                "val_r_label_loss_2": 0,
                "val_ppl":0
            }
    
    losses["val_r_loss"] = sum(val_result["loss"])/len(val_result["loss"])
    
    losses["val_r_recon_loss"] = sum(val_result["recon_loss"])/len(val_result["recon_loss"])
    
    losses["val_r_emb_loss"] = sum(val_result["emb_loss"])/len(val_result["emb_loss"])
    
    losses["val_r_label_loss"] = sum(val_result["label_loss"])/len(val_result["label_loss"])
    
    losses["val_r_label_loss_2"] = sum(val_result["label_loss_2"])/len(val_result["label_loss_2"])
    
    losses["val_ppl"] = sum(val_result["perplexity"])/len(val_result["perplexity"])
    
    return losses 

def get_reg_loss(result:dict, val_result:dict, ext_acc=None, val_ext_acc=None):
    """(reg) getting mean value of the result
    
    Args:
        result (dict): train results
        val_result (dict): validation results
        ext_acc (list): train accuracy
        val_ext_acc (list): validation accuracy

    Returns:
        mean value of each loss
    """
    
    losses = {
                "loss": 0,
                "val_loss": 0,
                "mae_acc": [],
                "val_mae_acc": []
            }
    
    losses["loss"] = sum(result["loss"])/len(result["loss"])
    losses["val_loss"] = sum(val_result["loss"])/len(val_result["loss"])
    
    if ext_acc is not None:
        losses["mae_acc"].append(ext_acc)
        losses["val_mae_acc"].append(val_ext_acc)
    else:
        losses["mae_acc"].append(np.mean(result["mae_acc"], axis=0).tolist())
        losses["val_mae_acc"].append(np.mean(val_result["mae_acc"], axis=0).tolist())
    
    return losses

def get_rec_loss(result:dict, val_result:dict):
    """(rec) getting mean value of the result
    Args:
        result (dict): train results
        val_result (dict): validation results

    Returns:
        mean value of each loss
    """
    
    losses = {  
                "r_loss": 0,
                "val_r_loss": 0,
                # "r_recon_loss": 0,
                # "val_r_recon_loss": 0,
                # "r_emb_loss": 0,
                # "val_r_emb_loss": 0,
                # "ppl": 0,
                # "val_ppl":0
            }
    
    losses["r_loss"] = sum(result["loss"])/len(result["loss"])
    losses["val_r_loss"] = sum(val_result["loss"])/len(val_result["loss"])
    
    # losses["r_recon_loss"] = sum(result["recon_loss"])/len(result["recon_loss"])
    # losses["val_r_recon_loss"] = sum(val_result["recon_loss"])/len(val_result["recon_loss"])
    
    # losses["r_emb_loss"] = sum(result["emb_loss"])/len(result["emb_loss"])
    # losses["val_r_emb_loss"] = sum(val_result["emb_loss"])/len(val_result["emb_loss"])
    
    # losses["ppl"] = sum(result["perplexity"])/len(result["perplexity"])
    # losses["val_ppl"] = sum(val_result["perplexity"])/len(val_result["perplexity"])
    
    return losses

@rank_zero_only
def print_info(losses: dict):
    """printing loss and accuracy"""
    
    print()
    print(f'[LOSS] train_loss | {losses["r_loss"]:.4f}    val_loss | {losses["val_r_loss"]:.4f}')
    print(f'[RECON LOSS] train_recon_loss | {losses["r_recon_loss"]:.4f}    val_recon_loss | {losses["val_r_recon_loss"]:.4f}')
    print(f'[EMBED LOSS] train_emb_loss | {losses["r_emb_loss"]:.4f}    val_emb_loss | {losses["val_r_emb_loss"]:.4f}')
    print(f'[LABEL LOSS] train_label_loss | {losses["r_label_loss"]:.4f}    val_label_loss | {losses["val_r_label_loss"]:.4f}')
    print(f'[LABEL LOSS] train_label_loss_2 | {losses["r_label_loss_2"]:.4f}    val_label_loss_2 | {losses["val_r_label_loss_2"]:.4f}')
    print()

def print_rec_info(losses: dict):
    """(reconstruction) printing loss and accuracy"""
    
    print()
    print(f'[LOSS] train_loss | {losses["r_loss"]:.4f}    val_loss | {losses["val_r_loss"]:.4f}')
    # print(f'[RECON LOSS] train_recon_loss | {losses["r_recon_loss"]:.4f}    val_recon_loss | {losses["val_r_recon_loss"]:.4f}')
    # print(f'[EMBED LOSS] train_emb_loss | {losses["r_emb_loss"]:.4f}    val_emb_loss | {losses["val_r_emb_loss"]:.4f}')
    # print(f'[PERPLEXITY] train_perplexity | {losses["ppl"]:.4f}    val_perplexity | {losses["val_ppl"]:.4f}')
    print()

def print_reg_info(losses: dict):
    """(regression) printing loss and mae score"""

    print()
    print(f'[LOSS] train_loss | {losses["loss"]:.4f}    val_loss | {losses["val_loss"]:.4f}')
    print(f'[TRAIN 1-MAE|ACC] OPN/C1| {losses["mae_acc"][0][0]:.4f}   CON/C2| {losses["mae_acc"][0][1]:.4f}   EXT/C3| {losses["mae_acc"][0][2]:.4f}   AGR/C4| {losses["mae_acc"][0][3]:.4f}   NEU/C5| {losses["mae_acc"][0][4]:.4f}')
    print(f'[TRAIN 1-MAE|ACC MEAN] {sum(losses["mae_acc"][0])/len(losses["mae_acc"][0])}')
    
    print(f'[VALID 1-MAE|ACC] OPN/C1| {losses["val_mae_acc"][0][0]:.4f}   CON/C2| {losses["val_mae_acc"][0][1]:.4f}   EXT/C3| {losses["val_mae_acc"][0][2]:.4f}   AGR/C4| {losses["val_mae_acc"][0][3]:.4f}   NEU/C5| {losses["val_mae_acc"][0][4]:.4f}')
    print(f'[VALID 1-MAE|ACC MEAN] {sum(losses["val_mae_acc"][0])/len(losses["val_mae_acc"][0])}')
    print()


def log(writer:SummaryWriter, tag:str, scalar_value:list, global_step=None, walltime=None, new_style=False, double_precision=False):
    """logging the output of the training process

    Args:
        writer (SummaryWriter): defined summarywriter
        tag (str): data identifier
        scalar_value (list): value to be saved
        global_step (_type_, optional): Defaults to None.
        walltime (_type_, optional): Defaults to None.
        new_style (bool, optional): Defaults to False.
        double_precision (bool, optional): Defaults to False.
    """
    # log in tensorboard
    writer.add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
    
    body = f"{tag}: {scalar_value:.4f}"
    prefix = f"Epoch {global_step + 1:03}| " if global_step is not None else ""
    
    # log in log
    logging.info(prefix + body)


def get_time_duration(start, end):
    sec = (end - start)
    result = str(datetime.timedelta(seconds=sec)).split('.')
    
    return result[0]


def get_requires_grad(path, model):
    if not Path(path).exists():
        os.makedirs(path)
    SAVE_GRAD_PATH = Path(path)
    
    freezed_list = []
    for name, p in model.named_parameters():
        freezed_list.append(name)
        freezed_list.append(f'{p.requires_grad}')
    
    with open(SAVE_GRAD_PATH / 'freezed.txt', 'w+') as file:
        file.write('\n'.join(freezed_list))


def get_mae(logits: np.array, target:np.array):
    """compute 1-mae score"""
    mae = np.abs(logits[0] - target) # (B, 5)
    mae_mean = np.mean(mae, axis=0) # (1, 5)
    
    return ModelOutput(
        mae=mae,
        mae_mean=mae_mean,
    )

def get_onehot(pred):
    new_pred = []
    for i in pred:
        if i == 0:
            new_pred.append([1.0, 0.0, 0.0, 0.0, 0.0])
        elif i == 1:
            new_pred.append([0.0, 1.0, 0.0, 0.0, 0.0])
        elif i == 2:
            new_pred.append([0.0, 0.0, 1.0, 0.0, 0.0])
        elif i == 3:
            new_pred.append([0.0, 0.0, 0.0, 1.0, 0.0])
        elif i == 4:
            new_pred.append([0.0, 0.0, 0.0, 0.0, 1.0])
    
    return new_pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_acc(logits: np.array, target:np.array) -> np.array:
    """compute values to compute accuarcy"""
    # logits: (B, 5)
    # target: (B, 5)

    pred = np.argmax(sigmoid(logits), axis=1) # (B,)
    target = np.argmax(target, axis=1) # (B,)
    
    return ModelOutput(
        pred=pred,
        target=target,
    )

def get_accuracy(pred:list, target:list) -> list:
    """compute accuarcy"""
    # pred: (N/B, B)
    # target: (N/B, B)
    
    # into tensor format
    pred = torch.tensor(np.array(pred), dtype=torch.int64).view(-1) # (N/B, B)
    target = torch.tensor(np.array(target), dtype=torch.int64).view(-1) # (N/B, B)
    
    metric = MulticlassAccuracy(average=None, num_classes=5)
    metric.update(pred, target)
    accuracy = (metric.compute()).numpy().tolist()
    
    return accuracy