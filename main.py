import time
from pathlib import Path
from sconf import Config

from utils import get_time_duration
from wrapper import train_model, test_model, demo_model, evaluate_model
from configs import parse_arguments, set_random_seed

from pytorch_lightning.utilities import rank_zero_only
#--------------------------------------------------#

def main(config):
    
    if config.mode == "train":
        
        rank_zero_only(print)(f"[START] TEDI Training Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        set_random_seed(2024, multi=config.multi)
        train_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        rank_zero_only(print)(f"[END] TEDI Training Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")
        
    elif config.mode == "test":
        print(f"[START] TEDI Testing Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        set_random_seed(2024, multi=config.multi)
        test_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] TEDI Testing Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")

    elif config.mode == "eval":
        print(f"[START] TEDI Evaluate Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        set_random_seed(2024, multi=config.multi)
        evaluate_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] TEDI Evaluate Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")
    
    else: # demo
        print(f"[START] TEDI Demo Begins ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        start = time.time()
        
        set_random_seed(2024, multi=config.multi)
        demo_model(config)
        
        end = time.time()
        end_time = get_time_duration(start, end)
        
        print(f"[END] TEDI Demo Complete ! | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Time {end_time}")

if __name__ == "__main__":
    
    args, config = parse_arguments()
    # set_random_seed(2024, multi=config.multi)
    main(config)