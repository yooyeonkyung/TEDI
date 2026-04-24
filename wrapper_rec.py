import os
import time
import torch
import logging
import torch.nn as nn
import pandas as pd
from evaluate import load
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from transformers import T5Tokenizer
from torch.utils.tensorboard import SummaryWriter

from configs_rec import parse_arguments, set_random_seed
from TEDI.model.tedi import MODEL

from dataset.dataset import get_train_valid_dataset, get_test_dataset
from utils import save_config_file, save_model_results, recon_results, get_time_duration
from utils import save_recon_results, print_rec_info, log, get_rec_loss
#-----------------------------------------------------------------------#


def get_recon_loss(logits, target, bos_tok_id):
    """compute reconstruction loss"""
    # logits: (B, L, V)
    # target: (B, L)
    # bos_tok_id: int (0)
    vocab = logits.shape[2]
    logits = logits.view(-1, vocab) # (B*L, V)
    target = target.view(-1) # (B*L)
    
    # use cross entropy loss
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=bos_tok_id)
    recon_loss = criterion(logits, target) # (B, V, L) / (B, L)
    
    return recon_loss


def train_model(config): 
    """train ted reconstruction model"""
    
    # make model save directory
    save_dir = f"{config.save_dir}/{config.datentime}"
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
                "model": config.model,
                # "emb_dim": 1024,
                # "emb_num": 10000,
                # "beta": 0.25
                }
    
    # save config
    save_config_file(config, save_dir, file_time, verbose=True)
    
    # set log
    logging.basicConfig(filename=Path(log_dir + f"/train_{file_time}.log"), level=logging.INFO, format="%(message)s")
    summary_writer = SummaryWriter(log_dir=log_dir)
    
    # get model
    # model = MODEL["T5_REC"](**model_params)
    model = MODEL["T5_REC_w_ADP"](**model_params)
    model.to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    
    result = {
        "n_updates": 0,
        # "recon_loss": [],
        # "emb_loss": [],
        "loss": [],
        # "perplexity": [],
    }
    
    val_result = {
        # "recon_loss": [],
        # "emb_loss": [],
        "loss": [],
        # "perplexity": [],
    }
    
    best_valid_loss = float("inf")
    
    for epoch in range(config.epochs):
        
        target = []
        recon = []
        val_target = []
        val_recon = []
        
        model.train()
        
        with tqdm(train_loader, desc=f"[TRAIN EPOCH {epoch+1}/{config.epochs}]") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                
                # tok_gen, logits, codebook, emb_loss, perplexity = model(config, x=batch)
                tok_gen, logits = model(config, x=batch)
                input_ids = batch["input_ids"].squeeze(1).to(config.device) # (B, L)
                
                # compute loss
                recon_loss = get_recon_loss(logits, input_ids, model.bos_token_id) # (B, L, V) / (B, L)
                # loss = recon_loss + emb_loss
                loss = recon_loss
                
                # aggregate generated sentence
                target, recon = recon_results(target, recon, input_ids, tok_gen)
                
                # ⭐⭐ grdient flows must be checked before training !!!
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                result["n_updates"] = i
                # result["recon_loss"].append(recon_loss.item())
                # result["emb_loss"].append(emb_loss.item())
                result["loss"].append(loss.item())
                # result["perplexity"].append(perplexity.item())
        
        with torch.no_grad():
            model.eval()
            with tqdm(valid_loader, desc=f"[VALID EPOCH {epoch+1}/{config.epochs}]") as tqdm_loader:
                for i, batch in enumerate(tqdm_loader):
                    
                    # val_tok_gen, logits, _, val_emb_loss, val_perplexity = model(config, x=batch)
                    val_tok_gen, logits = model(config, x=batch)
                    input_ids = batch["input_ids"].squeeze(1).to(config.device) # (B, L)
                    
                    # compute validation loss
                    val_recon_loss = get_recon_loss(logits, input_ids, model.bos_token_id)
                    # val_loss = val_recon_loss + val_emb_loss
                    val_loss = val_recon_loss
                    
                    # aggregate generated sentence
                    val_target, val_recon = recon_results(val_target, val_recon, input_ids, val_tok_gen)
                
                    # val_result["recon_loss"].append(val_recon_loss.item())
                    # val_result["emb_loss"].append(val_emb_loss.item())
                    val_result["loss"].append(val_loss.item())
                    # val_result["perplexity"].append(val_perplexity.item())

        # update and check learning rate
        print(f'[LR {epoch}] {scheduler.get_last_lr()}')
        scheduler.step()

        # save sentences (train)
        save_recon_results(save_dir, target, recon, epoch=epoch+1, name = "tr")
        
        # compute loss mean
        losses = get_rec_loss(result, val_result)
        
        # record log
        log(summary_writer, "loss/train", losses["r_loss"], epoch)
        log(summary_writer, "loss/valid", losses["val_r_loss"], epoch)
        
        # log(summary_writer, "recon_loss/train", losses["r_recon_loss"], epoch)
        # log(summary_writer, "recon_loss/valid", losses["val_r_recon_loss"], epoch)
        
        # log(summary_writer, "emb_loss/train", losses["r_emb_loss"], epoch)
        # log(summary_writer, "emb_loss/valid", losses["val_r_emb_loss"], epoch)
        
        # log(summary_writer, "perplexity/train", losses["ppl"], epoch)
        # log(summary_writer, "perplexity/valid", losses["val_ppl"], epoch)
        
        # save best validation model and result
        if losses["val_r_loss"] < best_valid_loss:
            
            # save model .pth file / codebook numpy file
            save_model_results(
                config, 
                path=save_dir,
                model=model, 
                epoch=epoch,
                val_loss=losses["val_r_loss"],
                best_loss=best_valid_loss, 
                result=result, 
                time=file_time, 
                verbose=config.verbose, 
                name='rec_model', 
                # codebook=codebook
                codebook=None
                )
            
            # save sentences (validation)
            save_recon_results(save_dir, val_target, val_recon, epoch=epoch+1, name="val")
            
            # best loss update
            best_valid_loss = losses["val_r_loss"]

        if i % config.log_interval == 0:
            print_rec_info(losses)
    
    summary_writer.flush()
    summary_writer.close()


def test_model(config):
    
    # get model saved path
    saved_dir = f"{config.save_dir}/{config.state}"
    print(f"[SAVED MODEL PATH] | {saved_dir}")
    
    # make test log directory
    test_dir = f"{saved_dir}/test"
    os.makedirs(test_dir, exist_ok=True)
    
    # set test log
    # logging.basicConfig(filename=Path(log_dir) / "test.log", level=logging.INFO, format="%(message)s")
    
    # check used device
    print(f"[DEVICE] | {config.device}")
    
    # prepare dataset
    test_loader = get_test_dataset(config) # len = 1997
    
    model_params = {
        "model": config.model,
        # "emb_dim": 1024,
        # "emb_num": 512,
        # "beta": 0.25,
    }
    
    # get model
    # model = MODEL["T5_REC"](**model_params)
    model = MODEL["T5_REC_w_ADP"](**model_params)
    model.load_state_dict(torch.load(
        Path(saved_dir) / "rec_model.pth", map_location=config.device
        ))
    model.to(config.device)

    result = {
        "x_gen": []
    }
    
    with torch.no_grad():
        model.eval()
        with tqdm(test_loader, desc=f"[TEST]") as tqdm_loader:
            for _, batch in enumerate(tqdm_loader):
                
                # reconstruction result
                output = model.generate(config, x=batch) # tok_gen, sent_gen
                result["x_gen"].append(output.sent_gen)
            
            result_sentence = pd.DataFrame(result["x_gen"], columns=["recon"])
            result_sentence.to_csv(f"{test_dir}/test_sentence.csv", index=False)


def main(config):
    
    if config.mode == 'train':
        
        print(f"[START] Reconstruction Train Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        train_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] Reconstruction Train Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")
        
    else:
        print(f"[START] Reconstruction Testing Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        test_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] Reconstruction Testing Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")


if __name__ == "__main__":
    
    args, config = parse_arguments()
    set_random_seed(2024, multi=config.multi)
    
    main(config)