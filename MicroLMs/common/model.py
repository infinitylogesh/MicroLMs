#%%
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

  def __init__(self,
               emb_dim,
               num_heads=8):
    super().__init__()
    self.num_heads = num_heads
    self.k_w = nn.Linear(emb_dim,emb_dim,bias=False)
    self.q_w = nn.Linear(emb_dim,emb_dim,bias=False)
    self.v_w = nn.Linear(emb_dim,emb_dim,bias=False)
    self.out = nn.Linear(emb_dim,emb_dim)
  
  def forward(self,x):

    batch_size,seq_len,emb_dim = x.size()
    head_dim = emb_dim // self.num_heads
    k = self.k_w(x)
    q = self.q_w(x)
    v = self.v_w(x)
    k = k.view(batch_size,seq_len,self.num_heads,head_dim)
    q = q.view(batch_size,seq_len,self.num_heads,head_dim)
    v = v.view(batch_size,seq_len,self.num_heads,head_dim)
    k = k.transpose(2,1).contiguous().view(batch_size*self.num_heads,seq_len,head_dim)
    q = q.transpose(2,1).contiguous().view(batch_size*self.num_heads,seq_len,head_dim)
    v = v.transpose(2,1).contiguous().view(batch_size*self.num_heads,seq_len,head_dim)

    dot = q@k.transpose(1,2)
    dot = F.softmax(dot, dim=2)
    attn = dot @ v
    attn = attn.view(batch_size,seq_len,self.num_heads*head_dim)

    return self.out(attn)

class TransformerBlock(nn.Module):

  def __init__(self,emb_dim,num_heads,seq_len,ff_dim,dropout=0.0):
    super().__init__()

    self.attention = SelfAttention(emb_dim,num_heads)
    self.norm1 = nn.LayerNorm(emb_dim)
    self.norm2 = nn.LayerNorm(emb_dim)
    self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim)
        )
    self.dropout = nn.Dropout(dropout)
  
  def forward(self,x):
    attn = self.attention(x)
    norm1 = self.norm1(attn+x)
    ff = self.ff(norm1)
    norm2 = self.norm2(ff+x)
    out = self.dropout(norm2)
    return out

class Transformer(nn.Module):

  def __init__(self,depth,vocab_size,emb_dim,num_heads,seq_len,ff_dim,dropout=0.0):
    super().__init__()
    self.token_embeddings = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim)
    self.pos_embeddings = nn.Embedding(num_embeddings=seq_len,embedding_dim=emb_dim)
    tfbs = [TransformerBlock(emb_dim,num_heads,seq_len,ff_dim,dropout) for _ in range(depth)]
    self.encoder = nn.Sequential(*tfbs)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self,x):
    tok = self.token_embeddings(x)
    batch_size,seq_len,emb_dim = tok.size()
    pos = self.pos_embeddings(torch.arange(seq_len))
    tok = tok + pos
    tok = self.dropout(tok)
    return self.encoder(tok)     

#%%
vocab_size = 100
seq_len =10
batch_size = 8
emb_dim = 128
tfb = Transformer(3,vocab_size,emb_dim,8,seq_len,4*emb_dim,0.2)

torch.arange(6)
x = torch.randint(0,100,(batch_size,seq_len))
tfb(x)