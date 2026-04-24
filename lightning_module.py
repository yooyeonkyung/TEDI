import os
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

## pytorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.plugins.io import CheckpointIO
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from model.tedi import MODEL

from dataset.dataset import (
    get_dataset,
    get_test_dataset,
    get_train_dataset,
    get_val_dataset,
)

from utils import (
    log,
    get_mae,
    get_acc,
    get_train_loss,
    get_val_loss,
    print_info,
    get_accuracy,
    get_requires_grad,
    save_model_results,    
)

class LightningTrainer(pl.LightningModule):
    def __init__(self, config, save_dir, model_params):
        super().__init__()
        self.config = config
        self.model = MODEL["T5_MODEL_S_w_ADP"](**model_params)
        self.load_and_freeze(self.model,
                    state1=(config.reg_d_state, 
                            "TED_Regression_Head",
                            "regression_head_dec"
                            ),
                    state2=(config.enc_s_state,
                            "encoder_R",
                            "encoder_s"
                            )
                    )

        self.save_dir = save_dir
        self.best_valid_loss = float("inf")
        self.automatic_optimization = False
    
    def freeze_all(self, model):
        """freeze model's parameters"""
        for p in model.parameters():
            p.requires_grad = False

    def load_state(self, state:torch.Tensor, module_name:str):
        """load trained state dicts"""
        load_dict = torch.load(state, map_location=self.config.device)
        state_dict = {k.replace(f'{module_name}.', ''): v for k, v in load_dict.items() if k.startswith(f'{module_name}.')}
        
        return state_dict

    def load_and_freeze(self, model, **kwargs):
        """load state and freeze the model
        
        - 241004
        [initialized list]
        : encoder_s / regression_head_dec
        [freezed list]
        : encoder_s / regression_head_dec
        
        """
        # state[0]: weight file path
        # state[1]: previous module name to get weight
        # state[2]: module name to assign weight
        # state[3]: whether it is a module
        for state in kwargs.values():
            state_dict = self.load_state(state[0], state[1])
            module = getattr(model, state[2])
            module.load_state_dict(state_dict)

        # freeze model
        # freeze_all(model.encoder_c)
        self.freeze_all(model.encoder_s)
        self.freeze_all(model.regression_head_dec)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr_m)
        optimizer_dec = optim.AdamW(self.model.parameters(), lr=self.config.lr_d)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=2)
        scheduler_dec = optim.lr_scheduler.ReduceLROnPlateau(optimizer_dec, 'min', factor=0.8, patience=2)
        return [optimizer, optimizer_dec], [scheduler, scheduler_dec]
    
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        path = Path(f"{self.config.save_dir}/{self.config.datentime}/tedi_model.pth")
        torch.save(checkpoint["state_dict"], path)
    
    def on_train_epoch_start(self):
        # code for debugging
        # torch.autograd.set_detect_anomaly(True)
        self.result = {
            "recon_loss": [],
            "emb_loss": [],
            "label_loss": [],
            "label_loss_2": [],
            "loss": [],
            "perplexity": [],
        }
        step = 20
        self.alpha = ((self.config.epochs/step) - (self.current_epoch//step)) / (self.config.epochs/step) # 1
        rank_zero_only(print)(f"[EPOCH {self.current_epoch+1} | ALPHA VALUE]: {self.alpha}")
        
        self.train_change_epoch = 9
        assert self.train_change_epoch != 0
        if self.config.debug == True:
            self.train_change_epoch = 0
        
        if self.current_epoch == self.train_change_epoch:
            rank_zero_only(print)("========== [STEP2] Training mechanism changed ==========")
    
    def training_step(self, batch, batch_idx):
        optimizer, optimizer_dec = self.optimizers()
        
        # train *0* origin
        output_dec_o = self.model(config=self.config, x_s=batch["batch_1"], x=batch["batch_1"], detach=True, dec=True)
        
        optimizer_dec.zero_grad()
        input_ids = batch["batch_1"]["input_ids"].squeeze(1) # (B, L)
        recon_loss_o = self.get_recon_loss(output_dec_o.logits, input_ids, self.model.bos_token_id) # (B, L, V) / (B, L)
        loss_r_o = (self.alpha)*recon_loss_o # loss (3)
        self.manual_backward(loss_r_o)
        optimizer_dec.step()
        
        output_o = self.model(config=self.config, x_s=batch["batch_1"], x=batch["batch_1"], dec=True)
        
        optimizer.zero_grad()
        loss_el_o = output_o.emb_loss + output_o.l_loss + output_o.l_loss_2 # loss (1) (2) (4) (5)
        self.manual_backward(loss_el_o)
        optimizer.step()
        
        if self.current_epoch >= self.train_change_epoch:
            
            # train *1* marginal
            output_m = self.model(config=self.config, x_s=batch["batch_2"], x=batch["batch_1"], dec=True)
            optimizer.zero_grad()
            loss_el_m = output_m.emb_loss + output_m.l_loss + output_m.l_loss_2
            self.manual_backward(loss_el_m)
            optimizer.step()
            
            batch_m = {
                "input_ids": torch.nan_to_num(output_m["tok_gen"]).unsqueeze(1), # (B, L)
                "attention_mask": torch.tensor([[1 if t!=0 else 0 for t in tok] for tok in output_m["tok_gen"]]).unsqueeze(1)
            }
            
            # train *2* cycle
            output_dec_c = self.model(config=self.config, x_s=batch["batch_1"], x=batch_m, detach=True)
            optimizer_dec.zero_grad()
            recon_loss_c = self.get_recon_loss(output_dec_c.logits, input_ids, self.model.bos_token_id) # (B, L, V)/(B, L)
            loss_r_c = (self.alpha)*recon_loss_c
            self.manual_backward(loss_r_c)
            optimizer_dec.step()
            
            output_c = self.model(config=self.config, x_s=batch["batch_1"], x=batch_m)
            optimizer.zero_grad()
            loss_el_c = output_c.emb_loss + output_c.l_loss + output_c.l_loss_2
            self.manual_backward(loss_el_c)
            optimizer.step()
            
            recon_loss = recon_loss_o + recon_loss_c
            emb_loss = output_o.emb_loss + output_m.emb_loss + output_c.emb_loss
            l_loss = output_o.l_loss + output_m.l_loss + output_c.l_loss
            l_loss_2 = output_o.l_loss_2 + output_m.l_loss_2 + output_c.l_loss_2
            loss = recon_loss + emb_loss + l_loss + l_loss_2
        else:
            recon_loss = recon_loss_o
            emb_loss = output_o.emb_loss
            l_loss = output_o.l_loss
            l_loss_2 = output_o.l_loss_2
            loss = recon_loss + emb_loss + l_loss + l_loss_2
        
        self.result["recon_loss"].append(recon_loss.item())
        self.result["emb_loss"].append((emb_loss).item())
        self.result["label_loss"].append((l_loss).item())
        self.result["label_loss_2"].append((l_loss_2).item())
        self.result["loss"].append((loss).item())
        self.result["perplexity"].append((output_o.perplexity).item())
    
    def on_train_epoch_end(self):
        losses = get_train_loss(self.result)
        
        self.log("step", self.current_epoch, sync_dist=True)
        self.log("loss/train", losses["r_loss"], prog_bar=True, sync_dist=True)
        self.log_dict({
            "recon_loss/train": losses["r_recon_loss"],
            "emb_loss/train": losses["r_emb_loss"],
            "label_loss/train": losses["r_label_loss"],
            "label_loss_2/train": losses["r_label_loss_2"],
            "perplexity/train": losses["ppl"]
            }, sync_dist=True
        )
    def on_validation_epoch_start(self):
        self.val_result = {
            "recon_loss": [],
            "emb_loss": [],
            "label_loss": [],
            "label_loss_2": [],
            "loss": [],
            "perplexity": [],
        }
        step = 20
        self.alpha = ((self.config.epochs/step) - (self.current_epoch//step)) / (self.config.epochs/step)
        
        self.train_change_epoch = 9
    
    def validation_step(self, batch, batch_idx):
        output_val = self.model(config=self.config, x_s=batch["batch_1"], x=batch["batch_1"])
        output_val_2 = self.model(config=self.config, x_s=batch["batch_2"], x=batch["batch_1"])
        
        input_ids = batch["batch_1"]["input_ids"].squeeze(1) # (B, L)
        val_recon_loss_1 = self.get_recon_loss(output_val.logits, input_ids, self.model.bos_token_id)
        val_recon_loss_2 = self.get_recon_loss(output_val_2.logits, input_ids, self.model.bos_token_id)

        if self.current_epoch >= self.train_change_epoch:
            val_recon_loss = val_recon_loss_1 + val_recon_loss_2
            val_emb_loss = output_val.emb_loss + output_val_2.emb_loss
            val_l_loss = output_val.l_loss + output_val_2.l_loss
            val_l_loss_2 = output_val.l_loss_2 + output_val_2.l_loss_2
        else:
            val_recon_loss = val_recon_loss_1
            val_emb_loss = output_val.emb_loss
            val_l_loss = output_val.l_loss
            val_l_loss_2 = output_val.l_loss_2
        
        val_loss = (self.alpha)*val_recon_loss + val_emb_loss + val_l_loss + val_l_loss_2
        
        self.val_result["recon_loss"].append(val_recon_loss.item())
        self.val_result["emb_loss"].append((val_emb_loss).item())
        self.val_result["label_loss"].append((val_l_loss).item())
        self.val_result["label_loss_2"].append((val_l_loss_2).item())
        self.val_result["loss"].append((val_loss).item())
        self.val_result["perplexity"].append((output_val.perplexity).item())
    
    def on_validation_epoch_end(self):
        losses = get_val_loss(self.val_result)
        
        # tensorboard / log
        self.log("step", self.current_epoch, sync_dist=True)
        self.log("loss/valid", losses["val_r_loss"], prog_bar=True, sync_dist=True)
        self.log_dict({
            "recon_loss/valid": losses["val_r_recon_loss"],
            "emb_loss/valid": losses["val_r_emb_loss"],
            "label_loss/valid": losses["val_r_label_loss"],
            "label_loss_2/valid": losses["val_r_label_loss_2"],
            "perplexity/valid": losses["val_ppl"]
            }, sync_dist=True
        )
        
        scheduler, scheduler_dec = self.lr_schedulers()
        scheduler.step(losses["val_r_label_loss"])
        scheduler_dec.step(losses["val_r_recon_loss"])
    
    def get_recon_loss(self, logits, target, bos_tok_id):
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


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def train_dataloader(self):
        self.train_set = get_train_dataset(self.config)
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4*self.config.gpu,
            pin_memory=True,
            drop_last=True
        )
        return train_loader
    
    def val_dataloader(self):
        self.val_set = get_val_dataset(self.config)
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4*self.config.gpu,
            pin_memory=True,
            drop_last=True
        )
        return val_loader


class ProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("[TRAIN] ")
        return bar
    
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("[VALID] ")
        return bar


class MyCustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "checkpoint.ckpt")
        state_dict = torch.load(path + "tedi_model.pth")
        checkpoint["state_dict"] = state_dict
        return checkpoint
    
    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)