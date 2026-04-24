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
    python wrapper_reg.py --mode train --case reg --dataset --epochs 5 --lr --device
    python wrapper_reg.py --mode train --case reg_d --dataset --epochs 5 --lr --device
    
    python wrapper_reg.py --mode train --case reg --dataset Yelp --epochs 20 --lr 0.0001 --device 0 --batch_size 64 (--softmax) --eos
    python wrapper_reg.py --mode train --case reg_d --dataset Yelp --epochs 20 --lr 0.0001 --device 0 --batch_size 64 (--softmax)
    
    [For Testing]
    python wrapper_reg.py --mode test --case reg --dataset --usage --device --batch_size 1 --t_date --t_time --eos
    python wrapper_reg.py --mode demo --case reg --dataset Amazon --usage tests --batch_size 1 --t_date 240805 --t_time 1437 --eos True
    python wrapper_reg.py --mode demo --case reg_d --dataset Yelp --usage tests --batch_size 1 --t_date 241024 --t_time 1435 --device 1
    
    python wrapper_reg.py --mode test --case reg_d --dataset --batch_size 1 --t_date --t_time
    python wrapper_reg.py --mode test --case reg_d --dataset Yelp --usage test --batch_size 1 --t_date 241024 --t_time 1435 --device 1
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--multi", dest="MULTI", type=bool, default=False,
                        help="multi-GPU to be used or not (default: %(default)s)")
    # parser.add_argument("--config", dest="CONFIG", type=str, default=None,
    #                     help="config file to be used")
    parser.add_argument("--t_date", dest="T_DATE", type=str, default=None,
                        help="date to be used for folder (default: %(default)s)")
    parser.add_argument("--t_time", dest="T_TIME", type=str, default=None,
                        help="time to be used for folder (default: %(default)s)")
    
    parser.add_argument("--device", dest="DEVICE", choices=[0, 1], type=int, default=0,
                        help="device to be used [0|1] (default: %(default)d)")
    parser.add_argument("--case", dest="CASE", choices=["reg", "reg_d"], type=str, required=True,
                        help="case of model [reg|reg_d]")
    parser.add_argument("--mode", dest="MODE", choices=["train", "test", "demo"], type=str, required=True,
                        help="run mode [train|test]")
    parser.add_argument("--dataset", dest="DATASET", choices=["Yelp", "Amazon", "FIV2", "FIV2_b"], type=str,
                        help="dataset to be used [Yelp|Amazon|FIV2|FIV2_b]")
    parser.add_argument("--debug", dest="DEBUG", type=bool, default=False,
                        help="check whether if it is debugging mode (default: %(default)s)")
    parser.add_argument("--sample", dest="SAMPLE", type=bool, default=False,
                        help="check whether if it is sampling mode (default: %(default)s)")
    parser.add_argument("--softmax", dest="SOFTMAX", type=bool, default=False,
                        help="case of cross entropy loss")
    parser.add_argument("--eos", dest="EOS", type=bool, default=False,
                        help="case of eos token reg/cls")
    parser.add_argument("--usage", dest="USAGE", type=str, default=None,
                        help="test data to be used (default: %(default)s)")
    
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
    args = parser.parse_known_args() # just the parameters inputted
    config = Configs(args[0]) # assign config to Configs
    
    return args, config


class Configs():
    def __init__(self, args):
        
        self.multi          = False
        self.model          = "google-t5/t5-large"
        
        # in case of test
        self.t_date         = "240610" if not args.T_DATE else args.T_DATE
        self.t_time         = "1550" if not args.T_TIME else args.T_TIME
        self.state          = f"{self.t_date}_{self.t_time}"
        
        self.device         = torch.device(f"cuda:{args.DEVICE}" if torch.cuda.is_available() else "cpu")
        self.mode           = args.MODE
        self.case           = args.CASE
        self.dataset        = args.DATASET
        self.max_length     = 128
        self.vocab_size     = 32128
        
        self.root           = os.getcwd()
        self.datentime      = time.strftime("%y%m%d_%H%M", time.localtime())
        self.date           = time.strftime("%y%m%d", time.localtime())
        self.time           = time.strftime("%H%M", time.localtime())
        self.save_dir       = f"{self.root}/result/{(self.dataset).lower()}_saved_{self.case}/{self.t_date}" if self.mode != "train" else f"{self.root}/result/{(self.dataset).lower()}_saved_{self.case}/{self.date}"
        self.epochs         = args.EPOCHS
        self.lr             = args.LEARNING_RATE
        self.batch_size     = args.BATCH_SIZE
        
        self.log_interval   = 1
        self.verbose        = False
        self.debug          = args.DEBUG
        self.sample         = args.SAMPLE
        self.softmax        = args.SOFTMAX
        self.eos            = args.EOS
        self.usage          = args.USAGE
        
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