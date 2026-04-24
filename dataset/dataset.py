import os
import torch
import random
import numpy as np
import pandas as pd
from sconf import Config
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer
#----------------------------------------------#


# dataset you want to use
path_dir = {
    "Amazon": {
        "per": "amazon_per.csv",
        "cos": "amazon_test_cosine.csv",
        # "train": "amazon_train_15.csv",
        # "valid": "amazon_valid_15.csv",
        # "test": "amazon_test_15.csv",
        "train": "amazon_train.csv",
        "valid": "amazon_valid.csv",
        "test": "amazon_test.csv",
        "tests": "amazon_testss.csv",
        "test1": "amazon_test1.csv",
        # "test1": "yelp_test1.csv", # for cross validation
        "test2": "amazon_test2.csv",
        "test3": "amazon_test3.csv",
        "test4": "amazon_test4.csv",
        "test5": "amazon_test5.csv",
        # "test5": "yelp_test5.csv", # for cross validation
        "sent1": "sentiment_test_1.csv",
        "sent5": "sentiment_test_5.csv"
    },
    "Yelp": {
        "per": "yelp_per.csv",
        "cos": "yelp_test_cosine.csv",
        # "train": "yelp_train_15.csv",
        # "valid": "yelp_valid_15.csv",
        # "test": "yelp_test_15.csv",
        "train": "yelp_train.csv",
        "valid": "yelp_valid.csv",
        "test": "yelp_test.csv",
        "tests": "yelp_testss.csv",
        "test1s": "yelp_test1s.csv",
        "test5s": "yelp_test5s.csv",
        "test1": "yelp_test1.csv",
        # "test1": "amazon_test1.csv", # for cross validation
        "test2": "yelp_test2.csv",
        "test3": "yelp_test3.csv",
        "test4": "yelp_test4.csv",
        "test5": "yelp_test5.csv",
        # "test5": "amazon_test5.csv", # for cross validation
        "sent1": "sentiment_test_1.csv",
        "sent5": "sentiment_test_5.csv"
    },
    "FIV2": {
        "train": "impression_train_drop.csv",
        "valid": "impression_validation.csv",
        "test": "impression_test_drop.csv",
    },
    "FIV2_b": {
        "train": "bin_train.csv",
        "valid": "bin_valid.csv",
        "test": "bin_test.csv",
    }
}

def get_train_valid_dataset(config):
    
    # personality dataset
    if config.dataset == "FIV2":
        if config.cycle:
            train_set = PdatasetTwo(config, usage="train")
            valid_set = PdatasetTwo(config, usage="valid")
        else:
            train_set = Pdataset(config, usage="train")
            valid_set = Pdataset(config, usage="valid")
    # sentiment dataset
    else:
        if config.cycle:
            train_set = SdatasetTwo(config, usage="train")
            valid_set = SdatasetTwo(config, usage="valid")
        else:
            train_set = Sdataset(config, usage="train")
            valid_set = Sdataset(config, usage="valid")
    
    train_loader = DataLoader(
                            train_set,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=4*config.gpu,
                            pin_memory=True,
                            drop_last=True
                        )
    valid_loader = DataLoader(
                            valid_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4*config.gpu,
                            pin_memory=True,
                            drop_last=True
                        )

    return train_loader, valid_loader

# for pytorch-lightning
def get_train_dataset(config):
    if config.dataset == "FIV2":
        if config.cycle:
            train_set = PdatasetTwo(config, usage="train")
        else:
            train_set = Pdataset(config, usage="train")
    else:
        if config.cycle:
            if config.debug:
                train_set = SdatasetTwo(config, usage="per")
            else:
                train_set = SdatasetTwo(config, usage="train")
        else:
            train_set = Sdataset(config, usage="train")
    return train_set

# for pytorch-lightning
def get_val_dataset(config):
    if config.dataset == "FIV2":
        if config.cycle:
            val_set = PdatasetTwo(config, usage="valid")
        else:
            val_set = Pdataset(config, usage="valid")
    else:
        if config.cycle:
            if config.debug:
                val_set = SdatasetTwo(config, usage="per")
            else:
                val_set = SdatasetTwo(config, usage="valid")
        else:
            val_set = Sdataset(config, usage="valid")
    return val_set

def get_test_dataset(config):
    
    # personality dataset
    if config.dataset == "FIV2":
        test_set = Pdataset(config, usage="test")
    # sentiment dataset
    else:
        test_set = Sdataset(config, usage=config.usage)
    
    test_loader = DataLoader(
                            test_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True
                        )
    
    return test_loader


def get_dataset(config, data=None):
    
    data_set = Tdataset(config, data)
    
    data_loader = DataLoader(
                            data_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True
    )
    
    return data_loader


# getting the path of the dataset used
def name_to_path(dataset_n, usage="train"):
    return os.path.join("data", dataset_n, path_dir[dataset_n][usage])


# custom dataset for any dataset (input_ids/attention_mask)
class Tdataset(Dataset):
    def __init__(self, config, data=None):
        
        self.tokenize = T5Tokenizer.from_pretrained(config.model, model_max_length=128, legacy=False)
        self.tf_data = data
        if data != None: 
            self.data = data
        else:
            self.data = pd.read_csv(f"{config.data_path}", index_col=False)
        self.max_len = config.max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if self.tf_data != None:
            self.token = self.tokenize(self.data[idx],
                            padding="max_length",
                            max_length=self.max_len,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt") # return in tensor type
        else:
            self.token = self.tokenize(self.data["text"][idx],
                                padding="max_length",
                                max_length=self.max_len,
                                truncation=True,
                                return_tensors="pt") # return in tensor type

        return {
                "input_ids": self.token["input_ids"],
                "attention_mask": self.token["attention_mask"],
                }


# custom dataset for personality dataset
class Pdataset(Dataset):
    def __init__(self, config, usage):
        
        # tokenizer
        self.tokenize = T5Tokenizer.from_pretrained(config.model, model_max_length=128, legacy=False)
        
        # dataset (data/label)
        data_dir = name_to_path(config.dataset, usage=usage)
        if config.sample == True:
            self.data = pd.read_csv(data_dir, index_col=False)[:config.batch_size*2] # test with sample
        else:
            self.data = pd.read_csv(data_dir, index_col=False)
        
        self.max_len = config.max_length
        
        self.label = self.data.iloc[:, -5:].to_numpy(dtype=np.float64)
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        self.token = self.tokenize(self.data["data"][idx],
                            padding="max_length",
                            max_length=self.max_len,
                            truncation=True,
                            return_tensors="pt") # return in tensor type

        return {
                "input_ids": self.token["input_ids"],
                "attention_mask": self.token["attention_mask"],
                "labels": torch.Tensor(self.label[idx])
                }

class PdatasetTwo(Dataset):
    def __init__(self, config, usage):
        
        # tokenizer
        self.tokenize = T5Tokenizer.from_pretrained(config.model, model_max_length=128, legacy=False)
        
        # dataset (data/label)
        data_dir = name_to_path(config.dataset, usage=usage)
        if config.sample == True:
            self.data = pd.read_csv(data_dir, index_col=False)[:config.batch_size*2] # test with sample
        else:
            self.data = pd.read_csv(data_dir, index_col=False)
        self.data_df = [self.data.iloc[i] for i in range(len(self.data))]
        self.max_len = config.max_length
        
        self.label = self.data.iloc[:, -5:].values
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        token_c = self.tokenize(self.data["data"].iloc[idx],
                            padding="max_length",
                            max_length=self.max_len,
                            truncation=True,
                            return_tensors="pt") # return in tensor type
        data_m = random.choice(self.data_df)
        token_m = self.tokenize(data_m["data"],
                            padding="max_length",
                            max_length=self.max_len,
                            truncation=True,
                            return_tensors="pt")
        
        return {
            "batch_1":
                {"input_ids": token_c["input_ids"],
                "attention_mask": token_c["attention_mask"],
                "labels": torch.Tensor(self.label[idx])},
            "batch_2":
                {"input_ids_m": token_m["input_ids"],
                "attention_mask_m": token_m["attention_mask"],
                "labels_m": torch.Tensor(data_m.iloc[-5:].to_numpy(dtype=np.float64))}
                }

# custom dataset for sentiment dataset (pos/neg)
class Sdataset(Dataset):
    def __init__(self, config, usage):
        
        # tokenizer
        self.tokenize = T5Tokenizer.from_pretrained(config.model, model_max_length=128, legacy=False)
        
        # dataset (data/label)
        data_dir = name_to_path(config.dataset, usage=usage)
        if config.sample == True:
            self.data = pd.read_csv(data_dir, index_col=False)[:config.batch_size*2]
        else:
            self.data = pd.read_csv(data_dir, index_col=False)
        
        self.max_len = config.max_length
        
        self.single_label = self.data.iloc[:, -6:-5].values
        self.label = self.data.iloc[:, -5:].values #.to_numpy(dtype=np.float64)
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        self.token = self.tokenize(self.data["text"][idx],
                            padding="max_length",
                            max_length=self.max_len,
                            truncation=True,
                            return_tensors="pt") # return in tensor type

        return {
                "input_ids": self.token["input_ids"],
                "attention_mask": self.token["attention_mask"],
                "single_labels": torch.tensor(self.single_label[idx], dtype=torch.int64),
                "labels": torch.Tensor(self.label[idx])
                }

# custom dataset for sentiment dataset (pos/neg) __version2__
class SdatasetTwo(Dataset):
    def __init__(self, config, usage):
        
        # tokenizer
        self.tokenize = T5Tokenizer.from_pretrained(config.model, model_max_length=128, legacy=False)
        
        # dataset (data/label)
        data_dir = name_to_path(config.dataset, usage=usage)
        if config.sample == True:
            # self.data = pd.read_csv(data_dir, index_col=False)[:config.batch_size*2]
            self.data = pd.read_csv("/home/ykyoo/yeonk/TED/data/Amazon/amazon_sample5.csv", index_col=False)
        else:
            self.data = pd.read_csv(data_dir, index_col=False)
        
        self.max_len = config.max_length
        
        self.single_label = self.data.iloc[:, -6:-5].values
        self.label = self.data.iloc[:, -5:].values
        
        self.data_by_label = {
            0.0: [self.data.iloc[i] for i in range(len(self.data)) if (self.single_label[i] == 0.0).item()],
            # 1.0: [self.data.iloc[i] for i in range(len(self.data)) if (self.single_label[i] == 1.0).item()],
            # 2.0: [self.data.iloc[i] for i in range(len(self.data)) if (self.single_label[i] == 2.0).item()],
            # 3.0: [self.data.iloc[i] for i in range(len(self.data)) if (self.single_label[i] == 3.0).item()],
            4.0: [self.data.iloc[i] for i in range(len(self.data)) if (self.single_label[i] == 4.0).item()]
        }
    
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        
        token_c = self.tokenize(self.data["text"].iloc[idx],
                            padding="max_length",
                            max_length=self.max_len,
                            truncation=True,
                            return_tensors="pt") # return in tensor type
        single_label_c = self.single_label[idx]
        other_labels = [l for l in self.data_by_label if l != single_label_c.item()]
        random_label = random.choice(other_labels)
        data_m = random.choice(self.data_by_label[random_label])
        token_m = self.tokenize(data_m["text"],
                            padding="max_length",
                            max_length=self.max_len,
                            truncation=True,
                            return_tensors="pt")
        
        return {
            "batch_1":
                {"input_ids": token_c["input_ids"],
                "attention_mask": token_c["attention_mask"],
                "single_labels": torch.tensor(self.single_label[idx], dtype=torch.int64),
                "labels": torch.Tensor(self.label[idx])},
            "batch_2":
                {"input_ids": token_m["input_ids"],
                "attention_mask": token_m["attention_mask"],
                "single_labels": torch.tensor([data_m["stars"]], dtype=torch.int64),
                "labels": torch.Tensor(data_m.iloc[-5:].to_numpy(dtype=np.float64))}
                }