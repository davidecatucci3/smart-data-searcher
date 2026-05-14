import torch.nn.functional as F
import torch.nn as nn
import torch

from transformers import ViTModel, BertModel
from hyperparameters import hyperparameters

# hyperparameters
d_i = hyperparameters['d_i']
d_t = hyperparameters['d_t']
d_e = hyperparameters['d_e']

# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # import pre-trained models for text encoder and image encoder
        self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", attn_implementation="sdpa") # sdpa a variant of flash attention
        self.txt_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased", attn_implementation="sdpa")

        # disable KV cache for memory saving
        self.img_encoder.config.use_cache = False
        self.txt_encoder.config.use_cache = False

        # save memory by not storing all gradients but this is slower due to recalculation after (USE ONLY IF YOU DON'T HAVE ENOUGH MEMORY)
        #self.img_encoder.gradient_checkpointing_enable()
        #self.txt_encoder.gradient_checkpointing_enable()
        
        # parameters
        self.W_i = nn.Parameter(torch.empty(d_i, d_e)) 
        self.W_t = nn.Parameter(torch.empty(d_t, d_e)) 
        self.t = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        
        # he initialization (kaiming) parameters -> sample weights from a uniform distribution ~ Unif(-sqrt(6/ fan_in), sqrt(6 / fan_in))
        torch.nn.init.kaiming_uniform_(self.W_i)
        torch.nn.init.kaiming_uniform_(self.W_t)

    def forward(self, I_batch, T_batch):
        # I_batch shape: [B, C, H, W]
        # T_batch shape: [B, max_length_txt]

        I_f = self.img_encoder(**I_batch).last_hidden_state # shape: [B, C*H*W, d_i]
        T_f = self.txt_encoder(**T_batch).last_hidden_state # shape: [B, max_length_txt, d_t]

        I_f = I_f[:, 0, :] # shape: [B, d_i]
        T_f = T_f[:, 0, :] # shape: [B, d_t]

        I_e = F.normalize(I_f @ self.W_i, p=2, dim=1) # shape: [B, d_e] -> normalize (makes it norm 1): rowi /= ||rowi||2
        T_e = F.normalize(T_f @ self.W_t, p=2, dim=1) # shape: [B, d_e] -> normalize (makes it norm 1): rowi /= ||rowi||2

        t_scale = torch.exp(self.t).clamp(max=100)

        logits = I_e @ T_e.T * t_scale # shape: [B, B] -> cosine similarity: just I_e @ T_e.T not divided by their norms because norms are 1

        return logits