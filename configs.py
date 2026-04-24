import os
import time
import torch
import random
import argparse
import numpy as np
from sconf import Config
#----------------#


def set_random_seed(random_seed, multi):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # in case of multi-GPU
    if multi == True:
        torch.cuda.manual_seed_all(random_seed)


def parse_arguments():
    """
    [For Training]
    python main.py --mode train --dataset FIV2 --epochs 20 --lr_1 0.0001 --lr_2 0.0001 / (batch_size|device|codebook|codebook_dim|embeds)
    python main.py --mode train --dataset Amazon --epochs 20 --batch_size 64 --codebook 1024 --codebook_dim 512 --embeds uniform
    
    [For Testing]
    python main.py --mode test --dataset FIV2 --batch_size 1 --t_date 240629 --t_time 1345 / (device)
    python main.py --mode test --dataset Amazon --batch_size 1 --t_date 240909 --t_time 2202 --device 0 --codebook 512 --codebook_dim 256 --embeds uniform --usage test1
    python main.py --mode test --dataset Amazon --batch_size 1 --t_date 240909 --t_time 2202 --device 0 --codebook 512 --codebook_dim 256 --embeds uniform --usage test1 --r 2
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--multi", dest="MULTI", type=bool, default=False,
                        help="multi-GPU to be used or not (default: %(default)s)")
    parser.add_argument("--t_date", dest="T_DATE", type=str, default=None,
                        help="date to be used for folder (default: %(default)s)")
    parser.add_argument("--t_time", dest="T_TIME", type=str, default=None,
                        help="time to be used for folder (default: %(default)s)")
    parser.add_argument("--device", dest="DEVICE", choices=[0, 1, 2, 3, 4, 5, 6, 7], type=int, default=0,
                        help="device to be used (default: %(default)d)")
    parser.add_argument("--mode", dest="MODE", choices=["train", "test", "eval", "demo"], type=str, required=True,
                        help="run mode [train|test|eval|demo]")
    parser.add_argument("--dataset", dest="DATASET", choices=["Yelp", "Amazon", "FIV2", "FIV2_b"], type=str,
                        help="dataset to be used [Yelp|Amazon|FIV2|FIV2_b]")
    parser.add_argument("--debug", dest="DEBUG", type=bool, default=False,
                        help="check whether if it is debugging mode (default: %(default)s)")
    parser.add_argument("--sample", dest="SAMPLE", type=bool, default=False,
                        help="check whether if it is sampling mode (default: %(default)s)")
    parser.add_argument("--softmax", dest="SOFTMAX", type=bool, default=False,
                        help="case of cross entropy loss (default: %(default)s)")
    parser.add_argument("--embeds", dest="EMBEDS", type=str, default=None,
                        help="embedding initialization to be used (default: %(default)s)")
    parser.add_argument("--usage", dest="USAGE", type=str, default=None,
                        help="test data to be used (default: %(default)s)")
    parser.add_argument("--option", dest="OPTION", type=str, default=None,
                    help="options for test dataset")
    parser.add_argument("--data_path", dest="DATA_PATH", type=str, default=None,
                    help="data path for evaluation")
    
    parser.add_argument("--codebook", dest="CODEBOOK", type=int, default=512,
                        help="number of codebook (default: %(default)d)")
    parser.add_argument("--codebook_dim", dest="CODEBOOK_DIM", type=int, default=1024,
                        help="dimension of  the codebook (default: %(default)d)")
    parser.add_argument("--alpha", dest="ALPHA", type=float, default=1.0,
                        help="alpha value to be used in reconstruction loss (default: %(default)f)")
    parser.add_argument("--beta", dest="BETA", type=float, default=1.0,
                        help="beta value to be used in label loss (default: %(default)f)")
    parser.add_argument("--r", dest="R", type=int, default=1,
                        help="r value to be used in codebook (default: %(default)d)")
    parser.add_argument("--zc", dest="ZC", type=int, default=1,
                        help="parameter of zc (default: %(default)d)")
    parser.add_argument("--cycle", dest="CYCLE", type=bool, default=False,
                        help="to use dataset for cycle loss (default: %(default)s)")
    
    parser.add_argument("--epochs", dest="EPOCHS", type=int, default=10, 
                        help="epochs to be used in 'train' mode (default: %(default)d)")
    parser.add_argument("--batch_size", dest="BATCH_SIZE", type=int, default=28, 
                        help="batch size to be used (default: %(default)d)")
    parser.add_argument("--lr_m", dest="LEARNING_RATE_MODEL", type=float, default=1e-04,
                        help="emb/label learning rate to be used in 'train' mode (default: %(default)f)")
    parser.add_argument("--lr_d", dest="LEARNING_RATE_DECODER", type=float, default=1e-05,
                    help="reconstruction learning rate to be used in 'train' mode (default: %(default)f)")

    # for multi-GPU
    parser.add_argument("--gpu", dest="GPU", type=int, default=None,
                        help="number of gpus using for training (default: %(default)d)")
    # parser.add_argument("--ws", dest="WORLD_SIZE", type=int, default=1,
    #                     help="number of nodes for distributed training")
    
    # config = parser.parse_args()
    args = parser.parse_known_args() # config, left_argv 얘는 입력 받은 애만
    config = Configs(args[0]) # 얜 Config
    
    return args, config


class Configs():
    def __init__(self, args):
        
        self.multi          = False
        
        self.model          = "google-t5/t5-large"
        self.dataset        = args.DATASET
        
        # in case of test
        self.t_date         = "240610" if not args.T_DATE else args.T_DATE
        self.t_time         = "1446" if not args.T_TIME else args.T_TIME
        self.state          = f"{self.t_date}_{self.t_time}"
        if self.dataset == "FIV2": # test용 reg 모델
            reg_date, reg_time = "240713", "1833"
            reg_state = f"{reg_date}_{reg_time}"
        elif self.dataset == "Amazon":
            reg_date, reg_time = "240807", "2048"
            reg_state = f"{reg_date}_{reg_time}"
        elif self.dataset == "Yelp":
            reg_date, reg_time = "241024", "1435"
            reg_state = f"{reg_date}_{reg_time}"
        self.reg_save_dir   = f"result/{(self.dataset).lower()}_saved_reg/{reg_date}/{reg_state}"
        
        self.mode           = args.MODE
        if self.mode == "train":
            self.device         = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device         = torch.device(f"cuda:{args.DEVICE}" if torch.cuda.is_available() else "cpu")
        self.max_length     = 128
        self.vocab_size     = 32128
        self.codebook       = args.CODEBOOK
        self.codebook_dim   = args.CODEBOOK_DIM
        self.alpha          = args.ALPHA
        self.beta           = args.BETA
        self.r              = args.R
        self.zc             = args.ZC
        self.cycle          = args.CYCLE
        
        self.root           = os.getcwd()
        self.datentime      = time.strftime("%y%m%d_%H%M", time.localtime())
        self.date           = time.strftime("%y%m%d", time.localtime())
        self.time           = time.strftime("%H%M", time.localtime())
        self.save_dir       = f"{self.root}/result/{(self.dataset).lower()}_saved_tedi/{self.t_date}" if self.mode != "train" else f"{self.root}/result/{(self.dataset).lower()}_saved_tedi/{self.date}"
        
        # for model freeze
        # self.rec_state      = "result/{(self.dataset).lower()}_saved/240711/240711_1643/rec_model.pth"
        if self.dataset == "FIV2": # 학습용 reg_d 모델
            reg_d_date, reg_d_time = "240708", "2023"
            reg_d_state = f"{reg_d_date}_{reg_d_time}"
        elif self.dataset == "Amazon":
            reg_d_date, reg_d_time = "240807", "2035"
            reg_d_state = f"{reg_d_date}_{reg_d_time}"
        elif self.dataset == "Yelp":
            reg_d_date, reg_d_time = "241024", "1435"
            reg_d_state = f"{reg_d_date}_{reg_d_time}"
        self.reg_d_state    = f"result/{(self.dataset).lower()}_saved_reg_d/{reg_d_date}/{reg_d_state}/reg_d_model.pth"
        
        if self.dataset == "FIV2": # eos token용 reg 모델
            enc_s_date, enc_s_time = "240724", "1653"
            enc_s_state = f"{enc_s_date}_{enc_s_time}"
        elif self.dataset == "Amazon":
            enc_s_date, enc_s_time = "240805", "1437"
            enc_s_state = f"{enc_s_date}_{enc_s_time}"
        elif self.dataset == "Yelp":
            enc_s_date, enc_s_time = "241025", "2153"
            enc_s_state = f"{enc_s_date}_{enc_s_time}"
        self.enc_s_state    = f"result/{(self.dataset).lower()}_saved_reg/{enc_s_date}/{enc_s_state}/reg_model.pth"
        
        self.epochs         = args.EPOCHS
        self.lr_m           = args.LEARNING_RATE_MODEL
        self.lr_d           = args.LEARNING_RATE_DECODER
        self.batch_size     = args.BATCH_SIZE
        
        self.log_interval   = 1
        self.verbose        = False
        self.debug          = args.DEBUG
        self.sample         = args.SAMPLE
        self.softmax        = args.SOFTMAX
        self.embeds         = args.EMBEDS
        self.usage          = args.USAGE
        self.option         = args.OPTION
        self.eos            = False
        self.data_path      = args.DATA_PATH
        
        self.gpu            = torch.cuda.device_count() if not args.GPU else args.GPU
        
        # for multi-GPU
        # if self.multi == True:
        #     self.world_size     = args.WORLD_SIZE * self.gpu
        #     self.rank           = 0


if __name__ == "__main__":
    
    # check values
    args, config = parse_arguments()
    print(args)
    print(config)
    print()
    # argparse values
    print(Config(args.__dict__).dumps())