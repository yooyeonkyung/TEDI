import os
import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from evaluate import load
from collections import OrderedDict
from transformers import T5Tokenizer
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# from model.tqvae import MODEL
from model.tedi import MODEL
from lightning_module import LightningTrainer, LightningDataModule, ProgressBar, MyCustomCheckpointIO

from dataset.dataset import (
    get_dataset,
    get_test_dataset,
    get_train_valid_dataset, 
)

from utils import (
    log,
    get_mae,
    get_acc,
    get_loss,
    print_info,
    get_accuracy,
    recon_results,
    save_config_file,
    get_requires_grad,
    save_recon_results,
    save_model_results,    
)
#----------------------------#

def get_checkpoint_info(path):
    ckpt = torch.load(f"{path}/checkpoint.ckpt")
    rank_zero_only(print)(f'[MODEL SAVED EPOCH]: {ckpt["epoch"]}')
    best_score = ckpt["callbacks"]["ModelCheckpoint{'monitor': 'label_loss/valid', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["best_model_score"]
    rank_zero_only(print)(f"[BEST MODEL SCore]: {best_score}")

def train_model(config): 
    """for model training
    check whether these are assigned adequately
    : self.reg_state
    """
    torch.set_float32_matmul_precision("high")
    
    # make model save directory
    save_dir = f"{config.save_dir}/{config.datentime}"
    os.makedirs(save_dir, exist_ok=True)
    
    # make log directory
    log_dir = f"{config.save_dir}/{config.datentime}/log"
    os.makedirs(log_dir, exist_ok=True)
    
    file_time = f"{config.time}"
    
    # check used device
    rank_zero_only(print)(f"[DEVICE] | {config.device}")
    
    model_params = {
        "config": config,
        "model": config.model,
        "emb_dim": config.codebook_dim,
        "emb_num": config.codebook,
        "beta": 0.25,
        "init_emb": None
        }
    
    # save config
    save_config_file(config, save_dir, file_time, verbose=True)
    
    # init model
    data_module = LightningDataModule(config)
    train_module = LightningTrainer(config, save_dir, model_params)
    
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=f"log",
        version=None,
        default_hp_metric=False
        )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="checkpoint",
        monitor="label_loss/valid", # can change this part
        save_top_k=1,
        save_last=False,
        mode="min"
        )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.gpu,
        strategy=DDPStrategy(find_unused_parameters=True),
        num_nodes=1,
        max_epochs=config.epochs,
        num_sanity_val_steps=0, # check valid step
        logger=logger,
        plugins=[MyCustomCheckpointIO()],
        callbacks=[checkpoint_callback, ProgressBar()]
        )
    trainer.fit(model=train_module, datamodule=data_module)
    
    get_checkpoint_info(save_dir)


def match_keys(load_weight: OrderedDict):
    new_keys = []
    with open("result/amazon_saved_tedi/keys.txt", "r") as f:
        for i in f:
            new_keys.append(i.strip())
    new_dict = OrderedDict()
    for i, (k, v) in enumerate(load_weight.items()):
        key = new_keys[i]
        new_dict[key] = v
    return new_dict

def test_model(config):
    """for model testing
    check whether these are assigned adequately
    : self.t_date / self.t_time (model trained)
    : self.reg_save_dir
    """
    
    # get model saved path
    saved_dir = f"{config.save_dir}/{config.state}"
    print(f"[SAVED MODEL PATH] | {saved_dir}")
    
    # make test log directory
    log_dir = f"{saved_dir}/test_log"
    os.makedirs(log_dir, exist_ok=True)
    
    # set test log
    logging.basicConfig(filename=Path(log_dir) / "test.log", level=logging.INFO, format="%(message)s")
    logging.info("-" * 60)
    logging.info(f"[TEST TIME] {config.datentime}")
    
    # check used device
    print(f"[DEVICE] | {config.device}")
    
    # prepare dataset
    test_loader = get_test_dataset(config)
    
    model_params = {
        "config": config,
        "model": config.model,
        "emb_dim": config.codebook_dim,
        "emb_num": config.codebook,
        "beta": 0.25,
        "init_emb": None
    }
    reg_model_params = {
        "config": config,
        "model": config.model
    }
    
    # get models
    model = MODEL["T5_MODEL_S_w_ADP"](**model_params)
    reg_model = MODEL["T5_REG_w_ADP"](**reg_model_params)
    
    model_weight = match_keys(torch.load(Path(saved_dir) / "tedi_model.pth", map_location=config.device))
    # weight = torch.load(Path(saved_dir) / "tedi_model.pth", map_location=config.device)
    model.load_state_dict(model_weight)
    model.to(config.device)
    
    reg_model.load_state_dict(torch.load(
        Path(config.reg_save_dir) / "reg_model.pth", map_location=config.device
    ))
    reg_model.to(config.device)
    
    x_inp = []
    x_rec = []
    x_gen = []
    rec_label = []
    gen_label = []
    
    # prepare style sentence
    # input_st_sent = input("[INPUT SENTENCE]: ")
    # input_st = input("[INPUT STYLE]: ")
    # input_st = list(map(float, input_st.split()))
    
    if config.dataset == "Amazon":
        if config.option == "test1":
            input_st_sent = "didn't like it at all."
            # input_st_sent = "this was one of the worst movies i have ever had the misfortune of enduring."
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
        elif config.option == "test2":
            input_st_sent = "but even they were bored."
            # input_st_sent = "i am truly disappointed with the final series."
            input_st = [0.0, 1.0, 0.0, 0.0, 0.0]
        elif config.option == "test3":
            input_st_sent = "well it could've been better than this."
            input_st = [0.0, 0.0, 1.0, 0.0, 0.0]
        elif config.option == "test4":
            # input_st_sent = "this was another good true story."
            input_st_sent = "i liked this one."
            # input_st_sent = "a great example of gilliam's directing."
            input_st = [0.0, 0.0, 0.0, 1.0, 0.0]
        elif config.option == "test5":
            input_st_sent = "great story and animation!"
            input_st = [0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            input_st_sent = "didn't like it at all."
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
    elif config.dataset == "Yelp":
        if config.option == "test1":
            # input_st_sent = "horrible experience and my carpet still looks terrible!"
            # input_st_sent = "i will never go here again!"
            input_st_sent = "it's so, so ,so, very, very, very bad."
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
        elif config.option == "test2":
            input_st_sent = "server not very good."
            # input_st_sent = "i won't go into detail, not very good."
            # input_st_sent = "not really worth watching."
            # input_st_sent = "not very exciting movie"
            input_st = [0.0, 1.0, 0.0, 0.0, 0.0]
        elif config.option == "test3":
            input_st_sent = "my friend, on the other hand, did enjoy it."
            # input_st_sent = "the foods were just okay."
            # input_st_sent = "overall, i think the garden is a nice affordable place to eat!"
            input_st = [0.0, 0.0, 1.0, 0.0, 0.0]
        elif config.option == "test4":
            # input_st_sent = "this place is pretty cool."
            input_st_sent = "room and bed was comfortable and clean."
            input_st = [0.0, 0.0, 0.0, 1.0, 0.0]
        elif config.option == "test5":
            # input_st_sent = "everything was seasoned perfectly, and we enjoyed everything."
            input_st_sent = "you guys are truly the best!!!"
            input_st = [0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            input_st_sent = "horrible experience and my carpet still looks terrible!"
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
    
    logging.info(f"[SENTENCE]: {input_st_sent}")
    logging.info(f"[INPUT STYLE]: {input_st}")

    tokenizer = T5Tokenizer.from_pretrained(config.model, model_max_length=128, legacy=False)
    # input_token: input_ids, attention_mask
    input_token = tokenizer(input_st_sent,
                            padding="max_length",
                            max_length=config.max_length,
                            truncation=True,
                            return_tensors="pt")
    batch_s = {"input_ids": input_token["input_ids"].unsqueeze(1),
                "attention_mask": input_token["attention_mask"].unsqueeze(1),
                "labels": torch.Tensor(input_st).reshape(1, 5)}
    
    with torch.no_grad():
        model.eval()
        with tqdm(test_loader, desc=f"[TEST]") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                
                inp_sent = tokenizer.decode(batch["input_ids"][0][0], skip_special_tokens=True)
                
                # print(f"[ORIGINAL SENTENCE] | {batch['labels']}") # (1, 5)
                rec_label.append(batch["labels"][0].numpy()) # (5,)
                gen_label.append(batch_s["labels"][0].numpy()) # (5,)
                
                # reconstruction result
                rec_tok = model.generate(config, x_s=batch, x=batch)
                rec_sent = tokenizer.decode(rec_tok[0], skip_special_tokens=True)
                # generation result
                gen_tok = model.generate(config, x_s=batch_s, x=batch)
                gen_sent = tokenizer.decode(gen_tok[0], skip_special_tokens=True)
                
                # gen_tok_re = model.generate(config, x_s=batch_s, x=batch, reverse=True)
                # gen_sent_re = tokenizer.decode(gen_tok_re[0], skip_special_tokens=True)
                
                x_inp.append(inp_sent)
                x_rec.append(rec_sent)
                x_gen.append(gen_sent)
                
                # for testing the code
                # if i == 2:
                #     break
    
    # save the test csv file
    x_rec_df = pd.DataFrame(x_rec, columns=["rec"])
    x_gen_df = pd.DataFrame(x_gen, columns=["gen"])
    df = pd.concat([x_rec_df, x_gen_df], axis=1)
    df.to_csv(Path(log_dir) / f"rec_gen_{list(config.usage)[-1]}to{torch.argmax(batch_s['labels']).item()+1}_{config.date}{config.time}.csv", index=False)
    # df.to_csv(f"result/amazon_saved_tedi/multi/rec_gen_{list(config.usage)[-1]}to{torch.argmax(batch_s['labels']).item()+1}_{config.date}{config.time}.csv", index=False)
    print("[SENTENCE SAVED]")
    
    # reconstruction result dataset
    x_rec_df["rec"] = x_rec_df["rec"].fillna("' '")
    x_gen_df["gen"] = x_gen_df["gen"].fillna("' '")
    x_rec = x_rec_df.rec.tolist()
    x_gen = x_gen_df.gen.tolist()
    
    rec_loader = get_dataset(config, x_rec)
    
    rec_mae_acc = []
    rec_acc_tgt = []
    
    with torch.no_grad():
        reg_model.eval()
        with tqdm(rec_loader, desc=f"[REC TEST]") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                
                logits = reg_model.inference(config, batch)
                if config.dataset == "FIV2":
                    output = get_mae(logits.detach().cpu().numpy(), np.expand_dims(rec_label[i], axis=0))
                    rec_mae_acc.append(output.mae)
                else:
                    output = get_acc(logits.detach().cpu().numpy(), np.expand_dims(rec_label[i], axis=0))
                    rec_mae_acc.append(output.pred)
                    rec_acc_tgt.append(output.target)
    
    if config.dataset == "FIV2":
        mae = np.array(rec_mae_acc)
        mae_mean = np.mean(mae, axis=0)
        logging.info(f"[REC/1-MAE]       | {mae.tolist()}")
        logging.info(f"[REC/1-MAE MEAN]  | {mae_mean.tolist()}")
    else:
        accuracy = get_accuracy(rec_mae_acc, rec_acc_tgt)
        logging.info(f"[REC/ACC]           | {accuracy}")
        logging.info(f"[REC/ACC MEAN]      | {sum(accuracy)/len(accuracy)}")
    
    # get bert score
    bertscore = load("bertscore")
    rec_score = bertscore.compute(predictions=x_rec, references=x_inp, model_type="distilbert-base-uncased")
    rec_f1 = sum(rec_score["f1"])/len(rec_score["f1"])
    logging.info(f"[REC/BERTScore]     | {rec_f1:.4f}")
    
    # generation result dataset
    gen_loader = get_dataset(config, x_gen)
    
    gen_mae_acc = []
    gen_acc_tgt = []
    
    with torch.no_grad():
        reg_model.eval()
        with tqdm(gen_loader, desc=f"[GEN TEST]") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                
                logits = reg_model.inference(config, batch)
                
                output = get_mae(logits.detach().cpu().numpy(), gen_label[i])
                if config.dataset == "FIV2":
                    output = get_mae(logits.detach().cpu().numpy(), np.expand_dims(gen_label[i], axis=0))
                    gen_mae_acc.append(output.mae)
                else:
                    output = get_acc(logits.detach().cpu().numpy(), np.expand_dims(gen_label[i], axis=0))
                    gen_mae_acc.append(output.pred)
                    gen_acc_tgt.append(output.target)
    
    logging.info("- " * 30)
    if config.dataset == "FIV2":
        mae = np.array(gen_mae_acc)
        mae_mean = np.mean(mae, axis=0)
        logging.info(f"[GEN/1-MAE]       | {mae.tolist()}")
        logging.info(f"[GEN/1-MAE MEAN]  | {mae_mean.tolist()}")
    else:
        accuracy = get_accuracy(gen_mae_acc, gen_acc_tgt)
        logging.info(f"[GEN/ACC]           | {accuracy}")
        logging.info(f"[GEN/ACC MEAN]      | {sum(accuracy)/len(accuracy)}")
    
    gen_score = bertscore.compute(predictions=x_gen, references=x_inp, model_type="distilbert-base-uncased")
    gen_f1 = sum(gen_score["f1"])/len(gen_score["f1"])
    logging.info(f"[GEN/BERTScore]     | {gen_f1:.4f}")


def evaluate_model(config):
    """for model testing
    check whether these are assigned adequately
    : self.t_date / self.t_time (model trained)
    : self.reg_save_dir
    """
    
    # get model saved path
    saved_dir = f"{config.save_dir}/{config.state}"
    print(f"[SAVED MODEL PATH] | {saved_dir}")
    
    # make test log directory
    log_dir = f"{saved_dir}/test_log"
    os.makedirs(log_dir, exist_ok=True)
    
    # set test log
    logging.basicConfig(filename=Path(log_dir) / "test_eval.log", level=logging.INFO, format="%(message)s")
    logging.info("-" * 60)
    logging.info(f"[TEST TIME] {config.datentime}")
    
    # check used device
    print(f"[DEVICE] | {config.device}")
    
    
    reg_model_params = {
        "config": config,
        "model": config.model
    }
    
    # get models
    reg_model = MODEL["T5_REG_w_ADP"](**reg_model_params)
    
    reg_model.load_state_dict(torch.load(
        Path(config.reg_save_dir) / "reg_model.pth", map_location=config.device
    ))
    reg_model.to(config.device)
    
    if config.option == "test1":
        input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
    elif config.option == "test2":
        input_st = [0.0, 1.0, 0.0, 0.0, 0.0]
    elif config.option == "test3":
        input_st = [0.0, 0.0, 1.0, 0.0, 0.0]
    elif config.option == "test4":
        input_st = [0.0, 0.0, 0.0, 1.0, 0.0]
    elif config.option == "test5":
        input_st = [0.0, 0.0, 0.0, 0.0, 1.0]
    else:
        input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
    
    if config.usage == "test":
        inp_data = pd.read_csv(f"data/{config.dataset}/{config.dataset.lower()}_{config.usage}.csv")
    elif config.usage == "sent1":
        inp_data = pd.read_csv(f"data/{config.dataset}/sentiment_test_1.csv")
    elif config.usage == "sent5":
        inp_data = pd.read_csv(f"data/{config.dataset}/sentiment_test_5.csv")
    else: 
        inp_data = pd.read_csv(f"data/{config.dataset}/{config.dataset.lower()}_{config.usage}.csv")
    
    x_inp = inp_data.text.tolist()
    rec_label = inp_data.iloc[:, -5:].to_numpy(dtype=np.float64).tolist()
    
    data = pd.read_csv(f"{config.data_path}", index_col=False)
    # data["rec"] = data["rec"].fillna("' '")
    data["gen"] = data["gen"].fillna("' '")
    # x_rec = data.rec.tolist()
    x_gen = data.gen.tolist()
    
    # define bertscore
    bertscore = load("bertscore")

    #-----------reconstruction result dataset-----------#
    # rec_loader = get_dataset(config, x_rec)
    
    # rec_mae_acc = []
    # rec_acc_tgt = []
    
    # with torch.no_grad():
    #     reg_model.eval()
    #     with tqdm(rec_loader, desc=f"[REC TEST]") as tqdm_loader:
    #         for i, batch in enumerate(tqdm_loader):
                
    #             logits = reg_model.inference(config, batch)
    #             if config.dataset == "FIV2":
    #                 output = get_mae(logits.detach().cpu().numpy(), np.expand_dims(rec_label[i], axis=0))
    #                 rec_mae_acc.append(output.mae)
    #             else:
    #                 output = get_acc(logits.detach().cpu().numpy(), np.expand_dims(rec_label[i], axis=0))
    #                 rec_mae_acc.append(output.pred)
    #                 rec_acc_tgt.append(output.target)
    
    # if config.dataset == "FIV2":
    #     mae = np.array(rec_mae_acc)
    #     mae_mean = np.mean(mae, axis=0)
    #     logging.info(f"[REC/1-MAE]       | {mae.tolist()}")
    #     logging.info(f"[REC/1-MAE MEAN]  | {mae_mean.tolist()}")
    # else:
    #     accuracy = get_accuracy(rec_mae_acc, rec_acc_tgt)
    #     logging.info(f"[REC/ACC]           | {accuracy}")
    #     logging.info(f"[REC/ACC MEAN]      | {sum(accuracy)/len(accuracy)}")
    
    # # get bert score
    # rec_score = bertscore.compute(predictions=x_rec, references=x_inp, model_type="distilbert-base-uncased")
    # rec_f1 = sum(rec_score["f1"])/len(rec_score["f1"])
    # logging.info(f"[REC/BERTScore]     | {rec_f1:.4f}")
    #-------------------------------------------------#
    
    # generation result dataset
    gen_loader = get_dataset(config, x_gen)
    
    gen_mae_acc = []
    gen_acc_tgt = []
    
    with torch.no_grad():
        reg_model.eval()
        with tqdm(gen_loader, desc=f"[GEN TEST]") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                
                logits = reg_model.inference(config, batch)
                
                output = get_mae(logits.detach().cpu().numpy(), input_st)
                if config.dataset == "FIV2":
                    output = get_mae(logits.detach().cpu().numpy(), np.expand_dims(input_st, axis=0))
                    gen_mae_acc.append(output.mae)
                else:
                    output = get_acc(logits.detach().cpu().numpy(), np.expand_dims(input_st, axis=0))
                    gen_mae_acc.append(output.pred)
                    gen_acc_tgt.append(output.target)
    
    logging.info("- " * 30)
    if config.dataset == "FIV2":
        mae = np.array(gen_mae_acc)
        mae_mean = np.mean(mae, axis=0)
        logging.info(f"[GEN/1-MAE]       | {mae.tolist()}")
        logging.info(f"[GEN/1-MAE MEAN]  | {mae_mean.tolist()}")
    else:
        accuracy = get_accuracy(gen_mae_acc, gen_acc_tgt)
        logging.info(f"[GEN/ACC]           | {accuracy}")
        logging.info(f"[GEN/ACC MEAN]      | {sum(accuracy)/len(accuracy)}")
    
    gen_score = bertscore.compute(predictions=x_gen, references=x_inp, model_type="distilbert-base-uncased")
    gen_f1 = sum(gen_score["f1"])/len(gen_score["f1"])
    logging.info(f"[GEN/BERTScore]     | {gen_f1:.4f}")


def demo_model(config):
    """for model demo
    check whether these are assigned adequately
    : self.t_date / self.t_time (model trained)
    : self.reg_save_dir
    """
    
    # get model saved path
    saved_dir = f"{config.save_dir}/{config.state}"
    print(f"[SAVED MODEL PATH] | {saved_dir}")
    
    # make test log directory
    log_dir = f"{saved_dir}/test_log"
    os.makedirs(log_dir, exist_ok=True)
    
    # set test log
    logging.basicConfig(filename=Path(log_dir) / "demo.log", level=logging.INFO, format="%(message)s")
    logging.info("-" * 60)
    logging.info(f"[DEMO TIME] {config.datentime}")
    
    # check used device
    print(f"[DEVICE] | {config.device}")
    
    # prepare dataset
    data_loader = get_dataset(config)
    
    model_params = {
        "config": config,
        "model": config.model,
        "emb_dim": config.codebook_dim,
        "emb_num": config.codebook,
        "beta": 0.25,
        "init_emb": None
    }
    
    # get models
    model = MODEL["T5_MODEL_S_w_ADP"](**model_params)
    
    model_weight = match_keys(torch.load(Path(saved_dir) / "tedi_model.pth", map_location=config.device))
    # weight = torch.load(Path(saved_dir) / "tedi_model.pth", map_location=config.device)
    model.load_state_dict(model_weight)
    model.to(config.device)
    
    x_inp = []
    x_rec = []
    x_gen = []
    rec_label = []
    gen_label = []
    
    if config.dataset == "Amazon":
        if config.option == "test1":
            input_st_sent = "didn't like it at all."
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
        elif config.option == "test2":
            input_st_sent = "but even they were bored."
            # input_st_sent = "i won't go into detail, not very good."
            # input_st_sent = "not really worth watching."
            # input_st_sent = "not very exciting movie"
            input_st = [0.0, 1.0, 0.0, 0.0, 0.0]
        elif config.option == "test3":
            input_st_sent = "well it could've been better than this."
            input_st = [0.0, 0.0, 1.0, 0.0, 0.0]
        elif config.option == "test4":
            input_st_sent = "i like the plot and the action and the villain works well, too."
            input_st = [0.0, 0.0, 0.0, 1.0, 0.0]
        elif config.option == "test5":
            input_st_sent = "great story and animation!"
            input_st = [0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            input_st_sent = "didn't like it at all."
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
    elif config.dataset == "Yelp":
        if config.option == "test1":
            input_st_sent = "horrible experience and my carpet still looks terrible!"
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
        elif config.option == "test2":
            input_st_sent = "server not very good."
            # input_st_sent = "i won't go into detail, not very good."
            # input_st_sent = "not really worth watching."
            # input_st_sent = "not very exciting movie"
            input_st = [0.0, 1.0, 0.0, 0.0, 0.0]
        elif config.option == "test3":
            input_st_sent = "my friend, on the other hand, did enjoy it."
            input_st = [0.0, 0.0, 1.0, 0.0, 0.0]
        elif config.option == "test4":
            input_st_sent = "this place is pretty cool."
            input_st = [0.0, 0.0, 0.0, 1.0, 0.0]
        elif config.option == "test5":
            input_st_sent = "everything was seasoned perfectly, and we enjoyed everything."
            input_st = [0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            input_st_sent = "horrible experience and my carpet still looks terrible!"
            input_st = [1.0, 0.0, 0.0, 0.0, 0.0]
    
    logging.info(f"[SENTENCE]: {input_st_sent}")
    logging.info(f"[INPUT STYLE]: {input_st}")

    tokenizer = T5Tokenizer.from_pretrained(config.model, model_max_length=128, legacy=False)
    # input_token: input_ids, attention_mask
    input_token = tokenizer(input_st_sent,
                            padding="max_length",
                            max_length=config.max_length,
                            truncation=True,
                            return_tensors="pt")
    batch_s = {"input_ids": input_token["input_ids"].unsqueeze(1),
                "attention_mask": input_token["attention_mask"].unsqueeze(1),
                "labels": torch.Tensor(input_st).reshape(1, 5)}
    
    with torch.no_grad():
        model.eval()
        with tqdm(data_loader, desc=f"[TEST]") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                
                # inp_sent = tokenizer.decode(batch["input_ids"][0][0], skip_special_tokens=True)
                
                # print(f"[ORIGINAL SENTENCE] | {batch['labels']}") # (1, 5)
                # rec_label.append(batch["labels"][0].numpy()) # (5,)
                # gen_label.append(batch_s["labels"][0].numpy()) # (5,)
                
                # reconstruction result
                # rec_tok = model.generate(config, x_s=batch, x=batch)
                # rec_sent = tokenizer.decode(rec_tok[0], skip_special_tokens=True)
                # generation result
                gen_tok = model.generate(config, x_s=batch_s, x=batch)
                gen_sent = tokenizer.decode(gen_tok[0], skip_special_tokens=True)
                
                # gen_tok_re = model.generate(config, x_s=batch_s, x=batch, reverse=True)
                # gen_sent_re = tokenizer.decode(gen_tok_re[0], skip_special_tokens=True)
                
                # x_inp.append(inp_sent)
                # x_rec.append(rec_sent)
                x_gen.append(gen_sent)
                
                # for testing the code
                # if i == 2:
                #     break
    
    # save the test csv file
    # x_rec_df = pd.DataFrame(x_rec, columns=["rec"])
    x_gen_df = pd.DataFrame(x_gen, columns=[f"gen_{torch.argmax(batch_s['labels']).item()+1}"])
    # df = pd.concat([x_rec_df, x_gen_df], axis=1)
    x_gen_df.to_csv(f"result/{(config.dataset).lower()}_saved_tedi/multi/demo_to{torch.argmax(batch_s['labels']).item()+1}_{config.date}{config.time}.csv", index=False)
    print("[SENTENCE SAVED]")