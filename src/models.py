"""
Models from the original paper
Copy this from your models.ipynb notebook
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pickle
import tqdm

class BinnedSeqDataset(Dataset):
    """Dataset class matching the original implementation"""
    def __init__(self, sequences, text_type=1):
        self.sequences = sequences
        self.text_type = text_type
    
    def __len__(self):
        return len(self.sequences)
    
    def __gen_text_rep__(self, text_dict):
        if self.text_type == 1:
            return text_dict["mean"]
        if self.text_type == 2:
            return torch.concat([text_dict["cls"], text_dict["mean"]], dim=-1)
    
    def __getitem__(self, idx):
        obs_feats = self.sequences[idx]
        
        res_feats = {}
        non_engage_rep = None
        engage_rep = None
        
        for k in obs_feats.keys():
            if ((k not in ["date_ranges", "username"]) and 
                ("text" not in k) and ("top" not in k)):
                
                if np.isscalar(obs_feats[k]):
                    res_feats[k] = torch.Tensor(np.array(obs_feats[k]).reshape(1, -1)).squeeze()
                else:
                    res_feats[k] = torch.Tensor(obs_feats[k])
            
            if "text" in k and str(self.text_type) in k:
                if "nengagement" in k:
                    non_engage_rep = self.__gen_text_rep__(obs_feats[k])
                else:
                    engage_rep = self.__gen_text_rep__(obs_feats[k])
        
        res_feats["text_feat"] = torch.concat([engage_rep, non_engage_rep], dim=-1)
        res_feats["date_ranges"] = [(x[0].strftime('%Y-%m-%d'), x[1].strftime('%Y-%m-%d')) 
                                   for x in obs_feats["date_ranges"]]
        res_feats["username"] = list(set(obs_feats["username"]))
        res_feats["hashtag_feat"] = obs_feats["tophashes"]
        res_feats["mentions_feat"] = obs_feats["topmentions"]
        
        return res_feats

class LSTMForecasterMF(nn.Module):
    """Multiple Feature Network - The main model from the paper"""
    
    def __init__(self, input_size, hidden_dim_lstm=64, hidden_text=128, 
                 hidden_inter=128, num_layers=1, bidirectional=False, 
                 dropout=0.1, output_size=7, activation="sigmoid"):
        super(LSTMForecasterMF, self).__init__()
        
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size + hidden_text + 4,
            hidden_size=hidden_dim_lstm,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc_text = nn.Linear(3072, hidden_text)
        self.fc_inter = nn.Linear(
            int(hidden_dim_lstm * self.num_directions) + 4 + 1536, 
            hidden_inter
        )
        self.fc_out = nn.Linear(hidden_inter, output_size)
    
    def forward(self, count_feats, in_time_feats, out_time_feats, 
                ns_feats, mention_feats, hash_feats, text_feats):
        
        self.lstm.flatten_parameters()
        
        text_feats = self.fc_text(text_feats)
        feats = torch.concat([count_feats, text_feats, in_time_feats], dim=-1)
        
        batch_size = feats.shape[0]
        out, (ht, ct) = self.lstm(feats)
        
        ht_reshaped = torch.reshape(ht, (self.num_layers, self.num_directions, batch_size, ht.shape[-1]))
        ht_final_layer = ht_reshaped[-1, :, :, :]
        
        if self.num_directions > 1:
            ht_forward = ht_final_layer[0, :, :]
            ht_backward = ht_final_layer[1, :, :]
            ht_final = torch.concat([ht_forward, ht_backward], dim=-1)
        else:
            ht_final = ht_final_layer[-1, :, :]
        
        inter_feats = torch.concat([ht_final, out_time_feats, hash_feats], dim=-1)
        fc_inter = self.activation(self.fc_inter(inter_feats))
        fc_out = self.fc_out(fc_inter)
        
        return fc_inter, fc_out