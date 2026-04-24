import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import ModelOutput
from transformers.models.t5.modeling_t5 import T5LayerNorm

"""
class:
    - ConcatAttention
    
    - FeatureAttention
    
    - AttentionBlock
    - SelfAttention
    - CrossAttention
    - StyleExtractionwAttention
    
"""

class ConcatAttention(nn.Module):
    """combining vectors w/concat attention"""
    def __init__(self, input_dim):
        super(ConcatAttention, self).__init__()
        self.fc = nn.Linear(input_dim*2, input_dim)
        self.v = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, z_sq, z_cq):
        # z_sq: (B, L, D)
        # z_cq: (B, L, D)
        batch_size, seq_len, hid_dim = z_sq.shape
        
        z_cat = torch.cat((z_sq, z_cq), dim=2) # (B, L, 2*D)
        # energy = torch.tanh(self.fc(z_cat.view(-1, 2*hid_dim))) # (B, L, 2*D)/(B, L, D)
        energy = torch.tanh(self.fc(z_cat)) # (B, L, 2*D)/(B, L, D)
        att_score = self.v(energy) # (B*L, 1)

        # return F.softmax(att_score.view(batch_size, seq_len, -1), dim=1)
        return F.softmax(att_score, dim=1)


class FABlock(nn.Module):
    """feature attention block"""
    def __init__(self, input_dim, r):
        super(FABlock, self).__init__()
        self.dense_z = nn.Sequential(
            nn.Linear(input_dim, input_dim//r),
            nn.Tanh()
        )
        self.dense_e = nn.Sequential(
            nn.Linear(input_dim//r, input_dim),
            nn.BatchNorm1d(input_dim, eps=1e-6, momentum=0.1, affine=True),
            nn.Sigmoid()
            )
    
    def forward(self, z, z_pool):
        z_out = self.dense_z(z_pool) # (B, 1, D//r)
        select_weights = self.dense_e(z_out.squeeze(1)) # (B, D)
        # select_weights_norm = torch.exp(select_weights)/torch.sum(torch.exp(select_weights)) # (B, D)
        # weighted_z = z * (select_weights_norm.unsqueeze(1))
        # weighted_z = z * (select_weights.unsqueeze(1))
        # weighted_z = z + (select_weights_norm.unsqueeze(1))
        weighted_z = z + (select_weights.unsqueeze(1))
        return weighted_z


class AttentionBlock(nn.Module):
    def __init__(self, config, residual=True):
        super(AttentionBlock, self).__init__()
        self.d_model = config.d_model
        
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)
        
        self.o = nn.Linear(self.d_model, self.d_model)
        
        self.residual = residual

    def forward(self, hidden_states, key=None, mask=None):
        batch_size, seq_len, hid_dim = hidden_states.shape
        x_input = hidden_states
        hid = hidden_states
        # x_input: (B, Lq, D) | (B*5, 3, D) | (B, 15, D)
        # hid: (B, Lq, D) | (B*5, 3, D) | (B, 15, D)
        # key: (B, Lk, D) | - | (B, 128, D)
        
        q = self.q(hid)
        if key is not None:
            k = self.k(key)
            v = self.v(key)
        else:
            k = self.k(hid)
            v = self.v(hid)
        # q: (B, Lq, D) | (B*5, 3, D) | (B, 15, D)
        # k: (B, Lk, D) | (B*5, 3, D) | (B, 128, D)
        # v: (B, Lv, D) | (B*5, 3, D) | (B, 128, D)
        
        weight = torch.matmul(q, k.permute(0, 2, 1)) * (int(hid_dim)**(-0.5))
        # weight: (B, Lq, Lk) | (B*5, 3, 3) | (B, 15, 128)
        
        if mask is not None:
            mask = mask.view(batch_size, -1, mask.shape[1])
            weight = weight.masked_fill(mask == 0, -1e10)
        
        a = F.softmax(weight, dim=-1)
        # a: (B, Lq, Lk) | (B*5, 3, 3) | (B, 15, 128)
        
        hid = torch.matmul(a, v)
        # hid: (B, Lq, D) | (B*5, 3, D) | (B, 15, D)
        
        hid_o = self.o(hid)
        # hid_o: (B, Lq, D) | (B*5, 3, D) | (B, 15, D)

        if self.residual:
            return x_input + hid_o
        else:
            return hid_o


class SelfAttention(nn.Module):
    """self attention module"""
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.self_attn = AttentionBlock(config, residual=True)
        
    def forward(self, hidden_states):
        normed_hidden_states = self.layer_norm(hidden_states) # (B*5, 3, D)
        self_attn_output = self.self_attn(normed_hidden_states) # (B*5, 3, D)
        
        return self_attn_output


class CrossAttention(nn.Module):
    """cross attention module"""
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.cross_attn = AttentionBlock(config, residual=True)
    
    def forward(self, hidden_states, key_value, attention_mask):
        normed_hidden_states = self.layer_norm(hidden_states)
        normed_key_value = self.layer_norm(key_value)
        cross_attn_output = self.cross_attn(normed_hidden_states, key=normed_key_value, mask=attention_mask)
        
        return cross_attn_output


class StyleExtractionBlock(nn.Module):
    """sentence style extraction block using attention"""
    def __init__(self, config, embedding):
        super(StyleExtractionBlock, self).__init__()
        
        self.self_attn_block = SelfAttention(config)
        self.cross_attn_block = CrossAttention(config)
        
        style_token_ids = [539, 655, 0, 975, 17819,  2936, 996, 8674, 0, 2065, 179, 655, 6567, 17, 20231]
        init_embeds = embedding.weight[style_token_ids] # (15 ,D)
        self.style_embedding = nn.Embedding.from_pretrained(init_embeds, freeze=False) # (15, D)
        self.idx = torch.LongTensor([[i for i in range(15)]])
        
        self.n_style = 5

    def forward(self, z_s, attention_mask, style_q=None):
        batch_size = z_s.shape[0] # (B, L, D)
        
        if style_q is None:
            style_embeds = self.style_embedding((self.idx).to(z_s.device)) # (1, 15, D)
            batch_size_s, _, hid_dim_s = style_embeds.shape # (1, 15, D)
            style_embeds = style_embeds.view(batch_size_s * self.n_style, -1, hid_dim_s) # (1*5, 3, D)
            
            st_out = self.self_attn_block(style_embeds) # (1*5, 3, D)
            
            st_out = st_out.view(batch_size_s, -1, hid_dim_s) # (1, 15, D)
            style_q = st_out.repeat(batch_size, 1, 1) # (B, 15, D)
        
        output = self.cross_attn_block(style_q, z_s, attention_mask) # (B, 15, D)
        
        return output


class StyleExtractionBlockP(nn.Module):
    """sentence style extraction block using attention"""
    def __init__(self, config, embedding):
        super(StyleExtractionBlockP, self).__init__()
        
        self.self_attn_block = SelfAttention(config)
        self.cross_attn_block = CrossAttention(config)
        
        style_token_ids = [539, 655, 0, 975, 17819,  2936, 996, 8674, 0, 2065, 179, 655, 6567, 17, 20231]
        init_embeds = embedding.weight[style_token_ids] # (15 ,D)
        self.style_embedding = nn.Embedding.from_pretrained(init_embeds, freeze=False) # (15, D)
        self.idx = torch.LongTensor([[i for i in range(15)]])
        
        self.n_style = 5

    def forward(self, z_s, labels_s, attention_mask, style_q=None):
        batch_size = z_s.shape[0] # (B, L, D)
        
        if style_q is None:
            style_embeds = self.style_embedding((self.idx).to(z_s.device)) # (1, 15, D)
            batch_size_s, _, hid_dim_s = style_embeds.shape # (1, 15, D)
            style_embeds = style_embeds.view(batch_size_s * self.n_style, -1, hid_dim_s) # (1*5, 3, D)
            
            st_out = self.self_attn_block(style_embeds) # (1*5, 3, D)
            
            st_out = st_out.view(batch_size_s, -1, hid_dim_s) # (1, 15, D)
            style_q = st_out.repeat(batch_size, 1, 1) # (B, 15, D)
            # labels_s: (B, 5)
            labels_s = labels_s.unsqueeze(-1).repeat(1, 1, 3).view(batch_size, -1, 1) # (B, 5, 3)/(B, 15, 1)
            style_q = torch.mul(labels_s, style_q) # (B, 15, D)
        
        output = self.cross_attn_block(style_q, z_s, attention_mask) # (B, 15, D)
        
        return output


class StyleExtractionModule(nn.Module):
    """sentence style extraction module"""
    def __init__(self, config, embedding, block_num=6):
        super(StyleExtractionModule, self).__init__()
        
        self.blocks = nn.ModuleList([StyleExtractionBlock(config, embedding)
                                    for _ in range(block_num)]) # 6 blocks
        # self.blocks = nn.ModuleList([StyleExtractionBlockP(config, embedding)
        #                             for _ in range(block_num)]) # 6 blocks
    
    def forward(self, z_s, label_s, attention_mask):
        for i, block in enumerate(self.blocks):
            if i == 0:
                style_out = block(z_s, attention_mask) # should input attention mask
                # style_out = block(z_s, label_s, attention_mask)
            else:
                style_out = block(z_s, attention_mask, style_q=style_out)
                # style_out = block(z_s, label_s, attention_mask, style_q=style_out)
        
        return style_out