import os
import time
import torch
import logging
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from sconf import Config
from transformers import T5Tokenizer
from torch.utils.tensorboard import SummaryWriter


from configs_reg import (
    parse_arguments,
    set_random_seed,
)
from model.tedi import MODEL

from dataset.dataset import (
    get_test_dataset,
    get_train_valid_dataset,    
)

from utils import (
    log,
    get_mae,
    get_acc,
    get_accuracy,
    get_reg_loss,
    print_reg_info,
    save_config_file,
    get_time_duration,
    save_model_results,
)
#-----------------------------------------------#

def freeze_all(model):
    """freeze model's parameters"""
    for p in model.parameters():
        p.requires_grad = False

def freeze(model, case):
    """freeze the model adding options"""

    freeze_all(model.encoder_R)
    if case == "reg_d":
        freeze_all(model.decoder_R)

def train_model(config): 
    
    # make model save directory
    save_dir = f"{config.save_dir}/{config.datentime}" # case: reg/reg_d
    os.makedirs(save_dir, exist_ok=True)
    
    # make log directory
    log_dir = f"{config.save_dir}/{config.datentime}/log"
    os.makedirs(log_dir, exist_ok=True)
    
    file_time = f"{config.time}"
    
    # check used device
    print(f"[DEVICE] | {config.device}")
    
    # prepare dataset
    train_loader, valid_loader = get_train_valid_dataset(config)
    
    model_params = {
        "config": config,
        "model": config.model
        }
    
    # save config
    save_config_file(config, save_dir, file_time, verbose=True)
    
    # set log
    logging.basicConfig(filename=Path(log_dir + f"/train_{file_time}.log"), level=logging.INFO, format="%(message)s")
    summary_writer = SummaryWriter(log_dir=log_dir)
    
    # get model / send to GPU
    if config.case == "reg":
        model_params["eos"] = config.eos # add eos option
        model = MODEL["T5_REG_w_ADP"](**model_params)
        if config.eos == True:
            logging.info("[MODEL] with <EOS> token")
        else:
            logging.info("[MODEL] with MAP(mean average pooling)")
    elif config.case == "reg_d":
        model = MODEL["T5_REG_D_w_ADP"](**model_params)
    
    # ⭐ check whether the model is freezed or not !!
    # freeze model [option]
    # freeze(model, config.case)
    
    model.to(config.device)
    
    # set training details
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=2)
    
    # set vacant loss
    result = {
        "n_updates": 0,
        "loss": [],
        "mae_acc": [],
        "acc_tgt": []
    }
    
    val_result = {
        "loss": [],
        "mae_acc": [],
        "acc_tgt": []
    }
    
    best_valid_loss = float("inf")

    for epoch in range(config.epochs):
        
        model.train()
        
        with tqdm(train_loader, desc=f"[TRAIN EPOCH {epoch+1}/{config.epochs}]") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                
                output = model(config, batch) # (B, 5)
                
                loss_mean = torch.mean(output.loss) # (B, 5)
                
                # ⭐⭐ grdient flows must be checked before training !!!
                loss_mean.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # compute mae/acc score
                if config.dataset == "FIV2":
                    mae_output = get_mae((output.logits).detach().cpu().numpy(), batch[f"labels"].numpy())
                    result["mae_acc"].append(1-mae_output.mae_mean)
                else:
                    acc_output = get_acc((output.logits).detach().cpu().numpy(), batch[f"labels"].numpy())
                    result["mae_acc"].append(acc_output.pred)
                    result["acc_tgt"].append(acc_output.target)
                
                result["n_updates"] = i
                result["loss"].append(loss_mean.item())
        
        with torch.no_grad():
            model.eval()
            with tqdm(valid_loader, desc=f"[VALID EPOCH {epoch+1}/{config.epochs}]") as tqdm_loader:
                for i, batch in enumerate(tqdm_loader):
                    
                    output = model(config, batch)
                    
                    # compute mae/acc score
                    if config.dataset == "FIV2":
                        mae_output = get_mae((output.logits).detach().cpu().numpy(), batch[f"labels"].numpy())
                        val_result["mae_acc"].append((1-mae_output.mae_mean)) # (1, 5)
                    else:
                        acc_output = get_acc((output.logits).detach().cpu().numpy(), batch[f"labels"].numpy())
                        val_result["mae_acc"].append(acc_output.pred)
                        val_result["acc_tgt"].append(acc_output.target)
                    
                    val_result["loss"].append(torch.mean(output.loss).item())

        
        if config.dataset == "FIV2":
            losses = get_reg_loss(result, val_result)
        else:
            # get accuracy
            accuracy = get_accuracy(result["mae_acc"], result["acc_tgt"]) # list: 5
            val_accuracy = get_accuracy(val_result["mae_acc"], val_result["acc_tgt"]) # list: 5
            
            losses = get_reg_loss(result, val_result, accuracy, val_accuracy)
        
        # record log
        log(summary_writer, "loss/train", losses["loss"], epoch) # also has logging(just about the loss)
        log(summary_writer, "loss/valid", losses["val_loss"], epoch)
        logging.info(f"Epoch {epoch + 1:03}| 1-MAE|ACC/train: {losses['mae_acc'][0]}")
        logging.info(f"Epoch {epoch + 1:03}| 1-MAE|ACC/train Mean: {sum(losses['mae_acc'][0])/len(losses['mae_acc'][0])}")
        logging.info(f"Epoch {epoch + 1:03}| 1-MAE|ACC/valid: {losses['val_mae_acc'][0]}")
        logging.info(f"Epoch {epoch + 1:03}| 1-MAE|ACC/valid Mean: {sum(losses['val_mae_acc'][0])/len(losses['val_mae_acc'][0])}")
        
        # save model and result
        if losses["val_loss"] < best_valid_loss:
            # save model .pth file
            save_model_results(config,
                                save_dir, 
                                model, 
                                epoch, 
                                losses["val_loss"], 
                                best_valid_loss, 
                                result, 
                                file_time, 
                                config.verbose, 
                                f"{config.case}_model")

            # best loss update
            best_valid_loss = losses["val_loss"]
        
        # update and check learning rate
        # print(f'[LR {epoch}] {scheduler.get_last_lr()}')
        print(f'[LR | {epoch}] {optimizer.param_groups[0]["lr"]}')
        scheduler.step(losses["val_loss"])

        if i % config.log_interval == 0:
            print_reg_info(losses)
    
    summary_writer.flush()
    summary_writer.close()


def test_model(config):
    
    # get model saved path
    saved_dir = f"{config.save_dir}/{config.state}"
    
    # make test log directory
    log_dir = f"{saved_dir}/test_log"
    os.makedirs(log_dir, exist_ok=True)
    
    # check used device
    print(f"[DEVICE] | {config.device}")
    
    # prepare dataset
    test_loader = get_test_dataset(config)
    
    model_params = {
        "config": config,
        "model": config.model
        }
    
    # set log / file time (current date and time)
    logging.basicConfig(filename=Path(log_dir) / "test.log", level=logging.INFO, format="%(message)s")
    
    # get model
    if config.case == "reg":
        model_params["eos"] = config.eos
        model = MODEL["T5_REG_w_ADP"](**model_params)
    elif config.case == "reg_d":
        model = MODEL["T5_REG_D_w_ADP"](**model_params)
        
    # load state dict
    model.load_state_dict(torch.load(
        Path(saved_dir + f'/{config.case}_model.pth'), map_location=config.device
        ))
    model.to(config.device)
    
    result = {
        "mae_acc": [],
        "acc_tgt": []
    }
    
    with torch.no_grad():
        model.eval()
        with tqdm(test_loader, desc=f"[TEST]") as tqdm_loader:
            for batch in tqdm_loader:
                
                logits = model.inference(config, batch) # (B, 5)
                
                # get 1-mae score (personality)
                if config.dataset == "FIV2":
                    mae_output = get_mae(logits.detach().cpu().numpy(), batch["labels"].numpy())
                    result["mae_acc"].append((1-mae_output.mae_mean))
                # get accuracy (sentiment)
                else:
                    acc_output = get_acc(logits.detach().cpu().numpy(), batch["labels"].numpy())
                    result["mae_acc"].append(acc_output.pred)
                    result["acc_tgt"].append(acc_output.target)
    
    if config.dataset == "FIV2":
        test_mae_acc = np.mean(result["mae_acc"], axis=0).tolist() # (1, 5)
    else:
        test_mae_acc = get_accuracy(result["mae_acc"], result["acc_tgt"])
    
    # record log
    logging.info(f"1-MAE|ACC/test: {test_mae_acc}")
    logging.info(f"1-MAE|ACC/test (mean): {sum(test_mae_acc)/len(test_mae_acc)}")
    logging.info(f" ")
    
    # print info
    print()
    print(f'[1-MAE|ACC] OPN/C1| {test_mae_acc[0]:.4f}   CON/C2| {test_mae_acc[1]:.4f}   EXT/C3| {test_mae_acc[2]:.4f}   AGR/C4| {test_mae_acc[3]:.4f}   NEU/C5| {test_mae_acc[4]:.4f}')
    print(f'[1-MAE|ACC MEAN] {sum(test_mae_acc)/len(test_mae_acc):.4f}')
    print()

def demo_model(config):
    
    # get model saved path
    saved_dir = f"{config.save_dir}/{config.state}"
    
    # make test log directory
    log_dir = f"{saved_dir}/test_log"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOG DIRECTORY] | {log_dir}")
    
    # check used device
    print(f"[DEVICE] | {config.device}")
    
    # prepare dataset
    test_loader = get_test_dataset(config)
    
    model_params = {
        "config": config,
        "model": config.model
        }
    
    # set log / file time (current date and time)
    # logging.basicConfig(filename=Path(log_dir) / "test.log", level=logging.INFO, format="%(message)s")
    
    # get model
    if config.case == "reg":
        model_params["eos"] = config.eos
        model = MODEL["T5_REG_w_ADP"](**model_params)
    elif config.case == "reg_d":
        model = MODEL["T5_REG_D_w_ADP"](**model_params)
        
    # load state dict
    model.load_state_dict(torch.load(
        Path(saved_dir + f'/{config.case}_model.pth'), map_location=config.device
        ))
    model.to(config.device)
    
    result = {
        "mae_acc": [],
        "acc_tgt": []
    }
    
    with torch.no_grad():
        model.eval()
        with tqdm(test_loader, desc=f"[TEST]") as tqdm_loader:
            for batch in tqdm_loader:
                
                logits = model.inference(config, batch) # (B, 5)
                
                # get 1-mae score (personality)
                if config.dataset == "FIV2":
                    mae_output = get_mae(logits.detach().cpu().numpy(), batch["labels"].numpy())
                    result["mae_acc"].append((1-mae_output.mae_mean))
                # get accuracy (sentiment)
                else:
                    acc_output = get_acc(logits.detach().cpu().numpy(), batch["labels"].numpy())
                    result["mae_acc"].append(acc_output.pred)
                    result["acc_tgt"].append(acc_output.target)
    
    pred = pd.DataFrame(result["mae_acc"], columns=["pred"])
    target = pd.DataFrame(result["acc_tgt"], columns=["target"])
    label = pd.concat([pred, target], axis=1)
    label.to_csv(f"{log_dir}/{config.dataset}_pred_result.csv")
    
    if config.dataset == "FIV2":
        test_mae_acc = np.mean(result["mae_acc"], axis=0).tolist() # (1, 5)
    else:
        test_mae_acc = get_accuracy(result["mae_acc"], result["acc_tgt"])
    
    # record log
    # logging.info(f"1-MAE|ACC/test: {test_mae_acc}")
    # logging.info(f"1-MAE|ACC/test (mean): {sum(test_mae_acc)/len(test_mae_acc)}")
    # logging.info(f" ")
    
    # print info
    print()
    print(f'[1-MAE|ACC] OPN/C1| {test_mae_acc[0]:.4f}   CON/C2| {test_mae_acc[1]:.4f}   EXT/C3| {test_mae_acc[2]:.4f}   AGR/C4| {test_mae_acc[3]:.4f}   NEU/C5| {test_mae_acc[4]:.4f}')
    print(f'[1-MAE|ACC MEAN] {sum(test_mae_acc)/len(test_mae_acc):.4f}')
    print()


def main(config):
    
    if config.mode == "train":
        
        print(f"[START] Regression({config.case}) Train Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        train_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] Regression({config.case}) Train Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")
        
    elif config.mode == "test":
        print(f"[START] Regression({config.case}) Testing Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        test_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] Regression({config.case}) Testing Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")
    
    elif config.mode == "demo":
        print(f"[START] Regression({config.case}) Demo Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        demo_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] Regression({config.case}) Demo Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")


if __name__ == "__main__":
    
    args, config = parse_arguments()
    set_random_seed(2024, multi=config.multi)
    
    main(config)