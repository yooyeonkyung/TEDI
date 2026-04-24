import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

from peft import (
    LoraConfig,
    get_peft_model
)
from transformers.utils import ModelOutput
from transformers.models.t5.modeling_t5 import T5ClassificationHead

from model.discretizer import (
    VectorDiscretizer, 
    FeatureRefineOne, 
    FeatureRefineTwo
)

from model.modules import StyleExtractionModule, FABlock
#-----------------------------------------------------------------#

"""
class:
    - TEDIRegressionHead
    - TEDIRegressionHeadDec
    - TEDIRegression
    - TEDIRegressionwAdapter
    - TEDIRegressionDec
    - TEDIRegressionDecwAdapter
    
    - TEDI    
    - TEDISentencewAdapter
    
    - TEDIReconstruction
    - TEDIReconstructionwAdapter
    
"""


class TEDIRegressionHead(nn.Module):
    """head for sentence-level regression tasks/w encoder"""
    def __init__(self, config, model):
        super(TEDIRegressionHead, self).__init__()
        t5config = T5Config.from_pretrained(model)
        t5config.num_labels = 5
        self.regression_head = T5ClassificationHead(t5config)
        
        if config.dataset == "FIV2":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, encoder_outputs, attention_mask, labels=None):
        # encoder_outputs(z_s): (B, L, D)
        # attention_mask: (B, 1, L)
        batch_size, seq_len, _ = encoder_outputs.shape
        attention_mask = attention_mask.view(batch_size, seq_len, -1) # (B, L, 1)
        e_out = (encoder_outputs*attention_mask).sum(dim=1) / (attention_mask.sum(dim=1) + 1e-8) # (B, D)
        
        logits = self.regression_head(e_out) # (B, 5)
        
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels) # (B, 5)
        
        return ModelOutput(
            logits=logits,
            loss=loss,
        )


class TEDIRegressionHeadDec(nn.Module):
    """head for sentence-level regression tasks/w decoder"""
    def __init__(self, config, model):
        super(TEDIRegressionHeadDec, self).__init__()
        t5config = T5Config.from_pretrained(model)
        t5config.num_labels = 5
        self.regression_head = T5ClassificationHead(t5config)
        
        if config.dataset == "FIV2":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            if config.softmax == False:
                self.criterion = nn.BCEWithLogitsLoss(reduction="none")
            else:
                self.criterion = nn.CrossEntropyLoss(reduction="none")
        
        self.eos_token_id = t5config.eos_token_id # 1
    
    def forward(self, input_ids, hidden_states, labels=None):
        # input_ids: (B, L)
        # hidden_states: (B, L, D)
        
        # get the position of <eos> token
        eos_mask = input_ids.eq(self.eos_token_id).to(input_ids.device) # (B, L)
        
        batch_size, _, hidden_dim = hidden_states.shape
        
        # regression with <eos> token
        sentence_representation = hidden_states[eos_mask, :].view(batch_size, -1, hidden_dim)[:, -1, :] # (B, 1, D)/(B, D)
        logits = self.regression_head(sentence_representation) # (B, 5)
        
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
            # loss = self.criterion(logits, labels.view(-1))
        
        return ModelOutput(
            logits=logits,
            loss=loss,
        )


class TEDIRegression(nn.Module):
    """sentence regression tasks/w encoder"""
    def __init__(self, config, model):
        super(TEDIRegression, self).__init__()
        self.encoder_R = T5ForConditionalGeneration.from_pretrained(model).encoder
        self.TED_Regression_Head = TEDIRegressionHead(config, model)
    
    def forward(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].to(config.device) # (B, 1, L) / broadcast
        labels = x["labels"].to(config.device)
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask) # (B, L, D)
        hidden_states = encoder_outputs.last_hidden_state # (B, L, D)
        output = self.TED_Regression_Head(hidden_states, attention_mask, labels) # (B, 5)
        
        return output
    
    def inference(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].to(config.device) # (B, 1, L) / broadcast
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask) # (B, L, D 1024)
        hidden_states = encoder_outputs.last_hidden_state ## dimension check
        output = self.TED_Regression_Head(hidden_states, attention_mask) # (B, 5)
        
        return output.logits


class TEDIRegressionwAdapter(nn.Module):
    """sentence regression tasks/w encoder"""
    def __init__(self, config, model, eos=False):
        super(TEDIRegressionwAdapter, self).__init__()
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none"
        )
        
        self.encoder_R = get_peft_model(
            T5ForConditionalGeneration.from_pretrained(model).encoder,
            lora_config
        )
        if eos:
            self.TED_Regression_Head = TEDIRegressionHeadDec(config, model)
        else:
            self.TED_Regression_Head = TEDIRegressionHead(config, model)
    
    def forward(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(1).to(config.device) # (B, L)
        if config.softmax == False:
            labels = x["labels"].to(config.device)
        else:
            labels = x["single_labels"].to(config.device)
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask) # (B, L, D)
        hidden_states = encoder_outputs.last_hidden_state # (B, L, D)
        
        if config.eos:
            output = self.TED_Regression_Head(input_ids, hidden_states, labels) # logits, loss
        else:
            output = self.TED_Regression_Head(hidden_states, attention_mask, labels) # (B, 5)
        
        return output
    
    def inference(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(1).to(config.device) # (B, L)
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask) # (B, L, D 1024)
        hidden_states = encoder_outputs.last_hidden_state
        
        if config.eos:
            output = self.TED_Regression_Head(input_ids, hidden_states)
        else:
            output = self.TED_Regression_Head(hidden_states, attention_mask) # (B, 5)
        
        return output.logits


class TEDIRegressionDec(nn.Module):
    """sentence regression tasks/w decoder"""
    def __init__(self, model):
        super(TEDIRegressionDec, self).__init__()
        self.encoder_R = T5ForConditionalGeneration.from_pretrained(model).encoder
        self.decoder_R = T5ForConditionalGeneration.from_pretrained(model).decoder
        self.TED_Regression_Head = TEDIRegressionHeadDec(config, model)
        
        config = T5Config.from_pretrained(model)
        self.bos_token_id = config.decoder_start_token_id # 0
    
    def shift_right(self, input_ids):
        # input_ids: (B, L)
        shifted_input = input_ids.new_zeros(input_ids.shape)
        shifted_input[..., 1:] = input_ids[..., :-1].clone()
        shifted_input[..., 0] = self.bos_token_id
        
        return shifted_input
    
    def forward(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(1).to(config.device)
        labels = x["labels"].to(config.device)
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        decoder_outputs = self.decoder_R(
            input_ids=self.shift_right(input_ids),
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states_d = decoder_outputs.last_hidden_state
        output = self.TED_Regression_Head(input_ids, hidden_states_d, labels)
        
        return output

    def inference(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(1).to(config.device)
        labels = x["labels"].to(config.device)
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask)
        hidden_states = encoder_outputs.last_hidden_state # (B, L, D)
        decoder_outputs = self.decoder_R(
            input_ids=self.shift_right(input_ids),
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states_d = decoder_outputs.last_hidden_state # (B, L, D)
        output = self.TED_Regression_Head(input_ids, hidden_states_d, labels)
        
        return output.logits


class TEDIRegressionDecwAdapter(nn.Module):
    """sentence regression tasks/w decoder /w adapter"""
    def __init__(self, config, model):
        super(TEDIRegressionDecwAdapter, self).__init__()
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none"
        )
        
        self.encoder_R = get_peft_model(
            T5ForConditionalGeneration.from_pretrained(model).encoder,
            lora_config
        )
        self.decoder_R = get_peft_model(
            T5ForConditionalGeneration.from_pretrained(model).decoder,
            lora_config
        )
        self.TED_Regression_Head = TEDIRegressionHeadDec(config, model)
        
        t5config = T5Config.from_pretrained(model)
        self.bos_token_id = t5config.decoder_start_token_id # 0
    
    def shift_right(self, input_ids):
        # input_ids: (B, L)
        shifted_input = input_ids.new_zeros(input_ids.shape)
        shifted_input[..., 1:] = input_ids[..., :-1].clone()
        shifted_input[..., 0] = self.bos_token_id
        
        return shifted_input
    
    def forward(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(1).to(config.device)
        labels = x["labels"].to(config.device)
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        decoder_outputs = self.decoder_R(
            input_ids=self.shift_right(input_ids),
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states_d = decoder_outputs.last_hidden_state
        output = self.TED_Regression_Head(input_ids, hidden_states_d, labels)
        
        return output

    def inference(self, config, x):
        input_ids = x["input_ids"].squeeze(1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(1).to(config.device)
        labels = x["labels"].to(config.device)
        
        encoder_outputs = self.encoder_R(input_ids, attention_mask)
        hidden_states = encoder_outputs.last_hidden_state # (B, L, D)
        decoder_outputs = self.decoder_R(
            input_ids=self.shift_right(input_ids),
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states_d = decoder_outputs.last_hidden_state # (B, L, D)
        output = self.TED_Regression_Head(input_ids, hidden_states_d, labels)
        
        return output.logits


class TEDI(nn.Module):
    """conditional sentence generation tasks"""
    def __init__(self, model, emb_dim, emb_num, beta, init_emb):
        super(TEDI, self).__init__()
        
        # define two models
        self.encoder_c = T5ForConditionalGeneration.from_pretrained(model).encoder # ***
        
        # "google-t5/t5-large"
        self.encoder_s = T5ForConditionalGeneration.from_pretrained(model).encoder
        self.decoder = T5ForConditionalGeneration.from_pretrained(model).decoder
        
        self.embedding = self.decoder.embed_tokens
        
        config = T5Config.from_pretrained(model)
        self.bos_token_id = config.decoder_start_token_id # 0
        self.eos_token_id = config.eos_token_id # 1
        self.pad_token_id = config.pad_token_id # 0
        
        self.lm_head = T5ForConditionalGeneration.from_pretrained(model).lm_head
        
        # quantize encoder output to a discrete latent space
        self.vector_discretize_s = VectorDiscretizer(emb_num, emb_dim, beta, init_emb, freeze=False)
        
        self.regression_head_dec = TEDIRegressionHeadDec(model) # ***
    
    def shift_right(self, input_ids):
        # input_ids: (B, L)
        shifted_input = input_ids.new_zeros(input_ids.shape)
        shifted_input[..., 1:] = input_ids[..., :-1].clone()
        shifted_input[..., 0] = self.bos_token_id
        
        return shifted_input

    def forward(self, config, x_s, x, z_sq_ex=None, verbose=False):
        
        # sentence_s
        input_ids_s = x_s["input_ids"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        attention_mask_s = x_s["attention_mask"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        labels_s = x_s["labels"].to(config.device) # (B, 5)
        
        # sentence_c
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device)
        
        # encoding process
        z_s = self.encoder_s(
            input_ids=input_ids_s,
            attention_mask=attention_mask_s,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_c = self.encoder_c(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            ) # ***
        
        z_s = z_s.last_hidden_state # (B, L, D)
        z_c = z_c.last_hidden_state # (B, L, D)
        
        # discretization process
        emb_loss, z_sq, codebook, perplexity = self.vector_discretize_s(config, z_s) # (B, L 128, D 1024)
        
        # prepare decoder input
        # input_ids: (B, L) | z_cq: (B, L, D)
        dec_embds = self.embedding(self.shift_right(input_ids)) # (B, L) -> (B, L, D)
        bos_embd = dec_embds[:, 0, :].unsqueeze(1) # (B, 1, D)
        dec_inp_embds = dec_embds[:, 1:, :] # (B, L-1 127, D)
        decoder_input_embds = torch.cat((bos_embd, z_c, dec_inp_embds), dim=1) # (B, 2*L = 1 + L + L-1, D) / 1 128 127
        
        decoder_attention_mask = self.shift_right(torch.cat((attention_mask, attention_mask), dim=1)) # (B, 2*L)
        
        # decoding process
        dec_out = self.decoder(
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=z_sq if z_sq_ex is None else z_sq_ex,
            encoder_attention_mask=attention_mask_s,
            inputs_embeds=decoder_input_embds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False
            )
        
        dec_hid = dec_out.last_hidden_state # (B, 2*L, D)
        dec_hid_sc, dec_hid_sent = dec_hid[:, :config.max_length, :], dec_hid[:, config.max_length:, :]
        
        # get label loss
        reg_output = self.regression_head_dec(
            config=config, 
            input_ids=input_ids,
            decoder_hid=dec_hid_sc, 
            labels=labels_s
            ) # (B, D) / ***
        label_loss = torch.mean(reg_output.loss) # mean loss
        
        # get generated sentence
        logits = self.lm_head(dec_hid_sent) # (B, L, D) -> (B, L, V 32128)
        tok_gen = torch.argmax(logits, dim=2) # (B, L)

        if verbose:
            print(f"initial data shape: {x.shape}")
            print(f"encoded data shape: {z_s.shape}/{z_c.shape}")
            print(f"discretized data shape: {z_sq.shape}")
            print(f"reconstructed data shape: {tok_gen.shape}")
        
        # return z_sq, tok_gen, logits, codebook, emb_loss, label_loss, perplexity
        return ModelOutput(
            z_sq=z_sq,
            tok_gen=tok_gen,
            logits=logits,
            codebook=codebook,
            emb_loss=emb_loss,
            l_loss=label_loss,
            perplexity=perplexity,
        )
    
    # get the generated sentence
    def generate(self, config, x_s, x):

        # sentence_s
        input_ids_s = x_s["input_ids"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        attention_mask_s = x_s["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # sentence you want to reconstruct
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # encode style sentence
        z_s = self.encoder_s(
            input_ids=input_ids_s,
            attention_mask=attention_mask_s,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_c = self.encoder_c(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            ) # ***
        
        z_s = z_s.last_hidden_state # (B, L, D)
        z_c = z_c.last_hidden_state
        
        # discretization process
        _, z_sq, _, _ = self.vector_discretize_s(config, z_s) # (B, L, D)
        
        # prepare decoder input
        tok_gen = torch.tensor(
            [[self.bos_token_id]]*config.batch_size,
            dtype=torch.long,
            device=config.device,
            ) #(B, 1)
        bos_embd = self.embedding(tok_gen) # (B, 1) -> (B, 1, D)
        decoder_input_embds = torch.cat((bos_embd, z_c), dim=1) # (B, 1+L, D) / 1 128
        
        decoder_attention_mask = torch.cat((torch.zeros_like(tok_gen.clone().detach()), attention_mask), dim=1) # (B, 1+L)
        for _ in range(config.max_length):
            
            dec_out = self.decoder(
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=z_sq,
                encoder_attention_mask=attention_mask_s,
                inputs_embeds=decoder_input_embds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False
            )
            
            dec_hid = dec_out.last_hidden_state # (B, 1+L, D 1024)
            ## check dec_hid shape
            logits = self.lm_head(dec_hid[:, -1, :]) # (B, D 1024) -> (B, V 32128)
            next_token_id = torch.argmax(logits, dim=1) # (B)
            
            # input embedding update (you also have to update attention mask)
            decoder_input_embds = torch.cat(
                (decoder_input_embds, self.embedding(next_token_id).unsqueeze(0)), dim=1
                ) # (B, L', D)
            
            # decoder attention mask update
            if next_token_id == self.pad_token_id:
                pad = torch.tensor(
                    [[self.pad_token_id]]*config.batch_size, 
                    dtype=torch.long,
                    device=config.device
                ) # (B, 1)
                decoder_attention_mask = torch.cat((decoder_attention_mask, pad), dim=1)
            else:
                non_pad = torch.tensor(
                    [[1]]*config.batch_size,
                    dtype=torch.long,
                    device=config.device
                ) # (B, 1)
                decoder_attention_mask = torch.cat((decoder_attention_mask, non_pad), dim=1)
            
            tok_gen = torch.cat([tok_gen, next_token_id.unsqueeze(0)], dim=-1) # (B, L)
            
            if next_token_id == self.eos_token_id:
                break
    
        return tok_gen
        

class TEDISentencewAdapter(nn.Module):
    """conditional sentence generation tasks with adapter"""
    def __init__(self, config, model, emb_dim, emb_num, beta, init_emb):
        super(TEDISentencewAdapter, self).__init__()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"], # ⭐
            lora_dropout=0.1,
            bias="none",
            )
        
        # self.encoder_c = T5ForConditionalGeneration.from_pretrained(model).encoder
        self.encoder_c = get_peft_model(
            T5ForConditionalGeneration.from_pretrained(model).encoder,
            lora_config
            )
        
        # self.encoder_s = T5ForConditionalGeneration.from_pretrained(model).encoder
        self.encoder_s = get_peft_model(
            T5ForConditionalGeneration.from_pretrained(model).encoder,
            lora_config
            )
        self.decoder = get_peft_model(
            T5ForConditionalGeneration.from_pretrained(model).decoder,
            lora_config
            )
        
        self.dec_embedding = self.decoder.embed_tokens
        
        t5config = T5Config.from_pretrained(model)
        
        self.bos_token_id = t5config.decoder_start_token_id # 0
        self.eos_token_id = t5config.eos_token_id # 1
        self.pad_token_id = t5config.pad_token_id # 0
        
        self.vector_discretize = FeatureRefineOne(
            config, 
            t5config, 
            emb_num, 
            emb_dim, 
            beta, 
            init_emb,
            # feat_emb=emb_dim//config.r, # featurerefinetwo
            )
        
        # self.fa_block = FABlock(t5config.d_model, r=config.r)
        
        # 1 regression enc model (trainable)
        t5config.num_labels = 5
        self.regression_head = T5ClassificationHead(t5config)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        # 2 regression dec model (freeze)
        self.regression_head_dec = TEDIRegressionHeadDec(config, model)
        # 3 cycle regression model (trainable)
        self.regression_head_dec_2 = TEDIRegressionHead(config, model)
        
        self.lm_head = T5ForConditionalGeneration.from_pretrained(model).lm_head
    
    def shift_right(self, input_ids):
        # input_ids: (B, L)
        shifted_input = input_ids.new_zeros(input_ids.shape)
        shifted_input[..., 1:] = input_ids[..., :-1].clone()
        shifted_input[..., 0] = self.bos_token_id
        
        return shifted_input

    def get_sentence_mean(self, enc_hid, attn_mask):
        # enc_hid: (B, L, D)
        # attn_mask: (B, L)
        batch_size, _, hidden_dim = enc_hid.shape
        
        # get mean value of sentence
        attn_mask = attn_mask.unsqueeze(-1) # (B, L, 1)
        masked_enc_hid = torch.mul(enc_hid, attn_mask) # (B, L, D)
        sentence_mean = (masked_enc_hid.sum(dim=1) / attn_mask.sum(dim=1)).view(batch_size, -1, hidden_dim)
        
        return sentence_mean
    
    def get_sentence_token(self, encoder_hid, input_ids):
        # encoder_hid: (B, L, D)
        # input_ids: (B, L)
        batch_size, _, hidden_dim = encoder_hid.shape
        
        # get the position of eos token
        eos_mask = input_ids.eq(self.eos_token_id).to(input_ids.device)
        
        # get setence feature
        sentence_token = encoder_hid[eos_mask, :].view(batch_size, -1, hidden_dim)[:, -1, :]
        
        return sentence_token

    def forward(self, config, x_s, x, detach=False, dec=False):
        
        # sentence_s
        input_ids_s = x_s["input_ids"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        attention_mask_s = x_s["attention_mask"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        labels_s = x_s["labels"].to(config.device) # (B, 5)
        
        # sentence_c
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device)
        
        # encoding process
        z_s = self.encoder_s(
            input_ids=input_ids_s,
            attention_mask=attention_mask_s,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_c = self.encoder_c(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        z_c = z_c.last_hidden_state # (B, L, D)
        
        # get feature extracted vector <eos> from the sentence
        z_s = self.get_sentence_token(z_s, input_ids_s) # (B, D)
        z_sq, emb_loss, codebook, perplexity = self.vector_discretize(z_s)
        label_loss_2 = torch.mean(self.criterion(self.regression_head(z_sq), labels_s)) # (B, 5)
        z_sq = z_sq.unsqueeze(1) # (B, 1, D)
        
        # prepare decoder input
        dec_embds = self.dec_embedding(self.shift_right(input_ids)) # (B, L) -> (B, L, D)
        bos_embd = dec_embds[:, 0, :].unsqueeze(1) # (B, 1, D)
        dec_inp_embds = dec_embds[:, 1:, :] # (B, L-1 127, D)
        decoder_input_embds = torch.cat((bos_embd, z_sq, dec_inp_embds), dim=1) # (B, L+1 = 1 + 1 + L-1, D)
        sq_mask = torch.ones(
            (z_sq.shape[0], z_sq.shape[1]), 
            dtype=torch.long,
            device=config.device
            ) # (B, 1)
        decoder_attention_mask = self.shift_right(torch.cat((sq_mask, attention_mask), dim=1)) # (B, 1 + 1 + L-1)
        
        # applying cosine similarity
        l2_weight = F.cosine_similarity(z_c, z_sq.detach(), dim=-1).unsqueeze(-1) # (B, L, 1)
        z_c = torch.mul((config.zc)*l2_weight, z_c) # (B, L, D)
        
        # decoding process
        dec_out = self.decoder(
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=z_c,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_input_embds.detach() if detach else decoder_input_embds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False
            )
        
        dec_hid = dec_out.last_hidden_state # (B, 1+L, D)
        dec_hid_sc, dec_hid_sent = dec_hid[:, :z_sq.shape[1], :], dec_hid[:, z_sq.shape[1]:, :] # (B, 1, D), (B, L, D)
        
        # 1 get label loss
        # dec_hid_sc_mean = dec_hid_sc.sum(dim=1) / dec_hid_sc.shape[1] # (B, 1024)
        # reg_logits = self.regression_head(dec_hid_sc_mean) # (B, 15, 5)
        # label_loss = torch.mean(self.MSE_loss(reg_logits, labels_s)) # mean loss
        
        # 2 get label loss
        if dec:
            reg_d_output = self.regression_head_dec(input_ids, dec_hid_sent, labels_s)
        else:
            reg_d_output = self.regression_head_dec_2(dec_hid_sent, attention_mask, labels_s) # nan오류
        label_loss = torch.mean(reg_d_output.loss)
        
        # get generated sentence
        logits = self.lm_head(dec_hid_sent) # (B, L, D) -> (B, L, V 32128)
        tok_gen = torch.argmax(logits, dim=-1) # (B, L)
        
        return ModelOutput(
            tok_gen=tok_gen,
            logits=logits,
            codebook=codebook,
            emb_loss=emb_loss,
            l_loss=label_loss,
            l_loss_2=label_loss_2,
            perplexity=perplexity,
            )
    
    # get the generated sentence
    def generate(self, config, x_s, x, reverse=None):

        # sentence_s
        input_ids_s = x_s["input_ids"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        attention_mask_s = x_s["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # sentence you want to reconstruct
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # encode style sentence
        z_s = self.encoder_s(
            input_ids=input_ids_s,
            attention_mask=attention_mask_s,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_c = self.encoder_c(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        z_c = z_c.last_hidden_state # (B, L, D)
        
        # get feature extracted vector <eos> from the sentence
        z_s = self.get_sentence_token(z_s, input_ids_s) # (B, D)
        z_sq, _, _, _ = self.vector_discretize(z_s)
        z_sq = z_sq.unsqueeze(1) # (B, 1, D)
        
        ## for checking
        # if reverse:
        #     z_sq = (-1) * z_sq
        
        # prepare decoder input
        tok_gen = torch.tensor(
            [[self.bos_token_id]]*config.batch_size,
            dtype=torch.long,
            device=config.device,
            ) # (B, 1)
        bos_embd = self.dec_embedding(tok_gen) # (B, 1) / (B, 1, D)
        decoder_input_embds = torch.cat((bos_embd, z_sq), dim=1) # (B, L', D)
        sq_mask = torch.ones(
            (z_sq.shape[0], z_sq.shape[1]), 
            dtype=torch.long,
            device=config.device
            )# (B, 1)
        decoder_attention_mask = torch.cat((torch.zeros_like(tok_gen.clone().detach()), sq_mask), dim=1) # (B, L')
        
        l2_weight = F.cosine_similarity(z_c, z_sq, dim=-1).unsqueeze(-1) # (B, L, 1)
        z_c = torch.mul((config.zc)*l2_weight, z_c) # (B, L, D)
        
        for _ in range(config.max_length):
            
            dec_out = self.decoder(
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=z_c,
                encoder_attention_mask=attention_mask,
                inputs_embeds=decoder_input_embds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False
                )
            
            dec_hid = dec_out.last_hidden_state # (B, L', D)
            logits = self.lm_head(dec_hid[:, -1, :]) # (B, D) -> (B, V 32128)
            next_token_id = torch.argmax(logits, dim=1) # (B,)
            
            decoder_input_embds = torch.cat(
                (decoder_input_embds, self.dec_embedding(next_token_id).unsqueeze(1)), dim=1
                ) # (B, L', D)
            
            # decoder attention mask update
            if next_token_id == self.pad_token_id:
                pad = torch.tensor(
                    [[self.pad_token_id]]*config.batch_size, 
                    dtype=torch.long,
                    device=config.device
                ) # (B, 1)
                decoder_attention_mask = torch.cat((decoder_attention_mask, pad), dim=1)
            else:
                non_pad = torch.tensor(
                    [[1]]*config.batch_size,
                    dtype=torch.long,
                    device=config.device
                ) # (B, 1)
                decoder_attention_mask = torch.cat((decoder_attention_mask, non_pad), dim=1)
            
            tok_gen = torch.cat([tok_gen, next_token_id.unsqueeze(1)], dim=-1) # (B, L)
            
            if next_token_id == self.eos_token_id:
                break
    
        return tok_gen
    
    def get_tsne(self, config, x_s, x):
        
        # sentence_s
        input_ids_s = x_s["input_ids"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        attention_mask_s = x_s["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # sentence you want to reconstruct
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # encode style sentence
        z_s = self.encoder_s(
            input_ids=input_ids_s,
            attention_mask=attention_mask_s,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_c = self.encoder_c(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        z_c = z_c.last_hidden_state # (B, L, D)
        attn_mask = attention_mask # (B, L, 1)
        # attn_len = attn_mask.sum(dim=1).item()
        # z_c = (z_c[:,:attn_len,:]).view(-1, z_s.shape[-1])
        
        # get feature extracted vector <eos> from the sentence
        z_s = self.get_sentence_token(z_s, input_ids_s) # (B, D)
        z_sq, z_sq3 = self.vector_discretize.get_tsne(z_s) # (B, D)
        
        return z_s, z_sq, z_sq3, z_c, attn_mask
    
    def get_heatmap(self, config, x_s, x):
        
        # sentence_s
        input_ids_s = x_s["input_ids"].squeeze(dim=1).to(config.device) # (B, 1, L) / (B, L)
        attention_mask_s = x_s["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # sentence you want to reconstruct
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # encode style sentence
        z_s = self.encoder_s(
            input_ids=input_ids_s,
            attention_mask=attention_mask_s,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_c = self.encoder_c(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        z_c = z_c.last_hidden_state # (B, L, D)
        
        z_s = self.get_sentence_token(z_s, input_ids_s) # (B, D)
        z_sq, _, _, _ = self.vector_discretize(z_s) # (B, D)
        l2_weight = F.cosine_similarity(z_c, z_sq.detach(), dim=-1).unsqueeze(-1) # (B, L, 1)
        
        return l2_weight


class TEDIReconstruction(nn.Module):
    """sentence reconstruction tasks"""
    # def __init__(self, model, emb_dim, emb_num, beta):
    def __init__(self, model):
        super(TEDIReconstruction, self).__init__()
        
        self.model = T5ForConditionalGeneration.from_pretrained(model) # "google-t5/t5-large" -> config
        # self.tokenizer = T5Tokenizer.from_pretrained(model, legacy=True)
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        
        self.bos_token_id = self.model.config.decoder_start_token_id # 0
        self.eos_token_id = self.model.config.eos_token_id # 1
        
        self.lm_head = self.model.lm_head
        
        # self.vector_discretization = VectorDiscretizer(emb_num, emb_dim, beta, init_emb=None, freeze=False)
    
    def shift_right(self, input_ids):
        # input_ids: (B, L)
        shifted_input = input_ids.new_zeros(input_ids.shape)
        shifted_input[..., 1:] = input_ids[..., :-1].clone()
        shifted_input[..., 0] = self.bos_token_id
        
        return shifted_input
    
    def forward(self, config, x, verbose=False):
        
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # encode style sentence
        z_s = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        
        # discretization process
        # loss, z_q, codebook, perplexity, _, _ = self.vector_discretization(config, z_s) # (B 14, L 128, D 1024)
        
        # aggregate 
        dec_out = self.decoder(
            input_ids=self.shift_right(input_ids),
            # encoder_hidden_states=z_q,
            encoder_hidden_states=z_s,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False
            )
        
        dec_hid = dec_out.last_hidden_state # (B, L, D)
        logits = self.lm_head(dec_hid) # (B, L, V 32128)
        tok_gen = torch.argmax(logits, dim=2) # (B, L)

        if verbose:
            print(f"initial data shape: {x.shape}")
            print(f"encoded data shape: {z_s.shape}")
            # print(f"discrete data shape: {z_q.shape}")
            print(f"reconstructed data shape: {tok_gen.shape}")
        
        return tok_gen, logits#, codebook, loss, perplexity
    
    # get generated sentence
    def generate(self, config, x):
        
        # sentence you want to reconstruct
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].to(config.device)
        
        # encode style sentence
        z_s = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        
        # discretization process
        # _, z_q, _, _, _, _ = self.vector_discretization(config, z_s) # (B, L, D)
        
        tok_gen = torch.tensor(
            [[self.bos_token_id]]*config.batch_size,
            dtype=torch.long,
            device=config.device,
            ) # (B, L 1)
        
        for _ in range(config.max_length):
            
            dec_out = self.decoder(
                input_ids=tok_gen,
                # encoder_hidden_states=z_q, # encoder output⭐
                encoder_hidden_states=z_s, # encoder output⭐
                encoder_attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False
                )
            
            dec_hid = dec_out.last_hidden_state # (B, L, D 1024)
            logits = self.lm_head(dec_hid[:, -1, :]) # (B, D 1024) -> (B, V 32128)
            next_token_id = torch.argmax(logits, dim=1) # (B)
            
            tok_gen = torch.cat([tok_gen, next_token_id.unsqueeze(0)], dim=-1) # (B, L)
            
            if next_token_id == self.eos_token_id:
                break
        
        sent_gen = self.tokenizer.decode(tok_gen.squeeze(0), skip_special_tokens=True)
    
        return ModelOutput(
            tok_gen=tok_gen,
            sent_gen=sent_gen
        )


class TEDIReconstructionwAdapter(nn.Module):
    """sentence reconstruction tasks with adapter"""
    def __init__(self, model):
        super(TEDIReconstructionwAdapter, self).__init__()
        
        T5_model = T5ForConditionalGeneration.from_pretrained(model) # "google-t5/t5-large" -> config
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"], # ⭐
            lora_dropout=0.1,
            bias="none",
            )
        
        self.encoder = get_peft_model(T5_model.encoder, lora_config)
        self.decoder = get_peft_model(T5_model.decoder, lora_config)
        
        self.bos_token_id = T5_model.config.decoder_start_token_id # 0
        self.eos_token_id = T5_model.config.eos_token_id # 1
        
        self.lm_head = T5_model.lm_head
        
        self.tokenizer = T5Tokenizer.from_pretrained(model, model_max_length=128, legacy=False)
        
        # self.vector_discretization = VectorDiscretizer(emb_num, emb_dim, beta, init_emb=None, freeze=False)
    
    def shift_right(self, input_ids):
        # input_ids: (B, L)
        shifted_input = input_ids.new_zeros(input_ids.shape)
        shifted_input[..., 1:] = input_ids[..., :-1].clone()
        shifted_input[..., 0] = self.bos_token_id
        
        return shifted_input
    
    def forward(self, config, x, verbose=False):
        
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].squeeze(dim=1).to(config.device) # (B, L)
        
        # encode style sentence
        z_s = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        
        # discretization process
        # loss, z_q, codebook, perplexity, _, _ = self.vector_discretization(config, z_s) # (B 14, L 128, D 1024)
        
        # aggregate 
        dec_out = self.decoder(
            input_ids=self.shift_right(input_ids),
            # encoder_hidden_states=z_q,
            encoder_hidden_states=z_s,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False
            )
        
        dec_hid = dec_out.last_hidden_state # (B, L, D)
        logits = self.lm_head(dec_hid) # (B, L, V 32128)
        tok_gen = torch.argmax(logits, dim=2) # (B, L)

        if verbose:
            print(f"initial data shape: {x.shape}")
            print(f"encoded data shape: {z_s.shape}")
            # print(f"discrete data shape: {z_q.shape}")
            print(f"reconstructed data shape: {tok_gen.shape}")
        
        return tok_gen, logits#, codebook, loss, perplexity
    
    # get generated sentence
    def generate(self, config, x):
        
        # sentence you want to reconstruct
        input_ids = x["input_ids"].squeeze(dim=1).to(config.device) # (B, L)
        attention_mask = x["attention_mask"].to(config.device)
        
        # encode style sentence
        z_s = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
            )
        
        z_s = z_s.last_hidden_state # (B, L, D)
        
        # discretization process
        # _, z_q, _, _, _, _ = self.vector_discretization(config, z_s) # (B, L, D)
        
        tok_gen = torch.tensor(
            [[self.bos_token_id]]*config.batch_size,
            dtype=torch.long,
            device=config.device,
            ) # (B, L 1)
        
        for _ in range(config.max_length):
            
            dec_out = self.decoder(
                input_ids=tok_gen,
                # encoder_hidden_states=z_q, # encoder output⭐
                encoder_hidden_states=z_s, # encoder output⭐
                encoder_attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False
                )
            
            dec_hid = dec_out.last_hidden_state # (B, L, D 1024)
            logits = self.lm_head(dec_hid[:, -1, :]) # (B, D 1024) -> (B, V 32128)
            next_token_id = torch.argmax(logits, dim=1) # (B)
            
            tok_gen = torch.cat([tok_gen, next_token_id.unsqueeze(0)], dim=-1) # (B, L)
            
            if next_token_id == self.eos_token_id:
                break
        
        sent_gen = self.tokenizer.decode(tok_gen.squeeze(0), skip_special_tokens=True)
    
        return ModelOutput(
            tok_gen=tok_gen,
            sent_gen=sent_gen
        )


MODEL = {
    "T5_REG": TEDIRegression,
    "T5_REG_w_ADP": TEDIRegressionwAdapter,
    "T5_REG_D": TEDIRegressionDec,
    "T5_REG_D_w_ADP": TEDIRegressionDecwAdapter,
    "T5_REC": TEDIReconstruction,
    "T5_REC_w_ADP": TEDIReconstructionwAdapter,
    "T5_MODEL": TEDI,
    "T5_MODEL_S_w_ADP": TEDISentencewAdapter,
}