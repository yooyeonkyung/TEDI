import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

class VectorDiscretizer(nn.Module):
    """
    Discrete process of TQVAE

    Args:
        emb_num (int): number of embeddings
        emb_dim (int): dimension of embedding
        beta (float): beta of commitment loss, beta * ||z_e(x)-sg[e]||^2
    """
    def __init__(self, config, emb_num: int, emb_dim:int, beta:float, init_emb: torch.tensor, freeze: bool) -> torch:
        super(VectorDiscretizer, self).__init__()
        
        self.emb_num    = emb_num
        self.emb_dim    = emb_dim
        self.beta       = beta
        
        if init_emb is not None:
            self.embedding  = nn.Embedding.from_pretrained(init_emb, freeze)
        elif config.embeds == "origin":
            self.embedding = nn.Embedding(self.emb_num, self.emb_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.emb_num, 1.0 / self.emb_num) # initialize
        elif config.embeds == "uniform":
            embedding = l2norm(uniform_init(self.emb_num, self.emb_dim))
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False) # learnable
    
    def ignore_pad(self, enc_hid, attn_mask):
        # enc_hid: (B, L, D)
        # attn_mask: (B, L)
        attn_mask = attn_mask.unsqueeze(-1) # (B, L, 1)
        masked_enc_hid = torch.mul(enc_hid, attn_mask)
        
        return masked_enc_hid
    
    # def forward(self, z, attention_mask):
    def forward(self, z):
        """
        Inputs the output z_s of the encoder and maps
        it into the discrete latent space

        z_s (continuous) -> z_q (discrete)
        
        B: batch size
        L: length of the seuqence
        D: dimension of the embedding (T5-Large: 1024)
        
        """
        # z.shape: (B, L, D)
        
        # flatten z
        z_flatten = z.view(-1, self.emb_dim) # (B*L, D) / 기존
        
        # (z_s - e)^2 = z^2 + e^2 - 2*z_e*e
        # distances from z to embeddings e (l2 norm ver: cosine similarity)
        d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flatten, self.embedding.weight.t()) # (B*L, emb_num)
        
        # find minimum distance embedding index
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # unsqueeze(1) -> (B*L, 1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.emb_num).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1) # (B*L, emb_num)
        
        # get discrete latent vector
        z_q = torch.matmul(
            min_encodings, self.embedding.weight).view(z.shape) # (B*L, D) -> (B, L, D)
        
        # loss for embedding
        # embedding loss: update codebook , commitment loss: update encoder
        loss = torch.mean((z.detach()-z_q)**2) + \
            self.beta * torch.mean((z-z_q.detach())**2) # encoder update only

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # utility of the codebook (=perplexity)
        # to check the quality of the codebook
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        return loss, z_q, self.embedding.weight, perplexity

class FeatureRefineOne(nn.Module):
    """feature refinement method 1"""
    def __init__(self, config, t5config, emb_num, emb_dim, beta, init_emb):
        super(FeatureRefineOne, self).__init__()
        
        self.vector_discretizer = VectorDiscretizer(
            config,
            emb_num,
            emb_dim,
            beta,
            init_emb,
            freeze=False
        )
        
        self.fc_1 = nn.Linear(t5config.d_model, t5config.d_model)
        self.fc_2 = nn.Linear(t5config.d_model, emb_dim)
        self.fc_3 = nn.Linear(emb_dim, t5config.d_model)
    
    def forward(self, z_s):
        
        z_s = self.fc_1(z_s) # (B, D)
        z_sent_s = self.fc_2(z_s) # (B, D*)
        
        # get discretized vector of the sentence
        emb_loss, z_sq, codebook, perplexity = self.vector_discretizer(z_sent_s) # (B, D) / (B, D/2)
        z_sq = self.fc_3(z_sq) # (B, D)
        
        return z_sq, emb_loss, codebook, perplexity
    
    def get_tsne(self, z_s):
        z_s = self.fc_1(z_s) # (B, D)
        z_sent_s = self.fc_2(z_s) # (B, D*)
        
        # get discretized vector of the sentence
        emb_loss, z_sq, codebook, perplexity = self.vector_discretizer(z_sent_s) # (B, D) / (B, D/2)
        z_sq3 = self.fc_3(z_sq) # (B, D)
        
        return z_sq, z_sq3


class FeatureRefineTwo(nn.Module):
    """feature refinement method 2"""
    def __init__(self, config, t5config, emb_num, emb_dim, beta, init_emb, feat_emb):
        super(FeatureRefineTwo, self).__init__()
        
        self.vector_discretizer = VectorDiscretizer(
            config,
            emb_num,
            emb_dim,
            beta,
            init_emb,
            freeze=False
        )
        
        self.fc_1 = nn.Linear(t5config.d_model, feat_emb)
        self.fc_2 = nn.Linear(feat_emb, t5config.d_model)
    
    def forward(self, z_s):
        # z_s: (B, D)
        z_sent = torch.tanh(self.fc_1(z_s)) # (B, D*)
        z_sent = torch.sigmoid(self.fc_2(z_sent)) # (B, D)
        z_sent_s = z_s * z_sent
        
        # get discretized vector of the sentence
        emb_loss, z_sq, codebook, perplexity = self.vector_discretizer(z_sent_s) # (B, D)
        
        return z_sq, emb_loss, codebook, perplexity