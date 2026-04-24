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
    python wrapper_rec.py --mode train --dataset FIV2 --lr 0.0001 --epochs 5
    
    [For Testing]
    python wrapper_rec.py --mode test --dataset FIV2 --batch_size 1 --t_date  --t_time 
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--multi", dest="MULTI", type=bool, default=False,
                        help="multi-GPU to be used or not (default: %(default)s)")
    parser.add_argument("--t_date", dest="T_DATE", type=str, default=None,
                        help="date to be used for folder (default: %(default)s)")
    parser.add_argument("--t_time", dest="T_TIME", type=str, default=None,
                        help="time to be used for folder (default: %(default)s)")
    parser.add_argument("--device", dest="DEVICE", choices=[0, 1], type=int, default=0,
                        help="device to be used [0|1] (default: %(default)d)")
    parser.add_argument("--mode", dest="MODE", choices=["train", "test"], type=str, required=True,
                        help="run mode [train|test]")
    parser.add_argument("--dataset", dest="DATASET", choices=["Yelp", "Amazon", "FIV2", "FIV2_b"], type=str,
                        help="dataset to be used [Yelp|Amazon|FIV2|FIV2_b]")
    parser.add_argument("--debug", dest="DEBUG", type=bool, default=False,
                        help="check whether if it is debugging mode (default: %(default)s)")
    
    parser.add_argument("--epochs", dest="EPOCHS", type=int, default=10, 
                        help="epochs to be used in 'train' mode (default: %(default)d)")
    parser.add_argument("--batch_size", dest="BATCH_SIZE", type=int, default=28, 
                        help="batch size to be used (default: %(default)d)")
    parser.add_argument("--lr", dest="LEARNING_RATE", type=float, default=1e-04,
                        help="learning rate to be used in 'train' mode (default: %(default)f)")
    
    # for multi-GPU
    parser.add_argument("--ws", dest="WORLD_SIZE", type=int, default=1,
                        help="number of nodes for distributed training")
    
    # config = parser.parse_args()
    args = parser.parse_known_args() # config, left_argv 얘는 입력 받은 애만
    config = Configs(args[0]) # 얜 Config
    
    return args, config


class Configs():
    def __init__(self, args):
        
        self.multi          = False
        
        self.model          = "google-t5/t5-large"
        
        # for testing process
        self.t_date         = "240623" if not args.T_DATE else args.T_DATE
        self.t_time         = "1727" if not args.T_TIME else args.T_TIME
        self.state          = f"{self.t_date}_{self.t_time}"
        
        self.device         = torch.device(f"cuda:{args.DEVICE}" if torch.cuda.is_available() else "cpu")
        self.mode           = args.MODE
        self.dataset        = args.DATASET
        self.max_length     = 128
        self.vocab_size     = 32128
        
        self.root           = os.getcwd()
        self.datentime      = time.strftime("%y%m%d_%H%M", time.localtime())
        self.date           = time.strftime("%y%m%d", time.localtime())
        self.time           = time.strftime("%H%M", time.localtime())
        self.save_dir       = f"{self.root}/saved/{self.t_date}" if self.mode == "test" else f"{self.root}/saved/{self.date}"
        
        self.epochs         = int(10) if not args.EPOCHS else args.EPOCHS
        self.lr             = 1e-04 if not args.LEARNING_RATE else args.LEARNING_RATE
        self.batch_size     = 32 if not args.BATCH_SIZE else args.BATCH_SIZE
        
        self.log_interval   = 1
        self.verbose        = False
        self.debug          = False if not args.DEBUG else args.DEBUG
        
        self.gpu            = torch.cuda.device_count() # 2
        
        # for multi-GPU
        if self.multi == True:
            self.world_size     = args.WORLD_SIZE * self.gpu
            self.rank           = 0


if __name__ == "__main__":
    
    # check values
    args, config = parse_arguments()
    print(args)
    print(config)
    print()
    # argparse values
    print(Config(args.__dict__).dumps())