import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 100
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.0

with open("sources/encoder_datas_for_ai.txt", 'r', encoding='utf-8') as f:
    text = f.read()

strings = text.split("\n")

# Tokenizer işlemi
tokenizer = Tokenizer()
tokenizer.fit_on_texts(strings)
stoi = tokenizer.word_index
itos = dict(zip(stoi.values(), stoi.keys()))
vocab_size = len(stoi)+1

# Metni sayılara çevirme ve padding
sequences = tokenizer.texts_to_sequences(strings)
padsequences = pad_sequences(sequences, maxlen=block_size, padding='pre')

# Tensor formatına çevirme (flatten ile tek boyuta indirgeme)
data_en = torch.tensor(padsequences, dtype=torch.long).flatten()



from torch.utils.data import Dataset


"""class MultiLabelDataset(Dataset):
  def __init__(self,data):
    self.data=data
  def __getitem__(self, idx):
        x = self.data[idx:idx + block_size + 1]
        
        # Eğer boyut yetmiyorsa sıfırlarla doldur
        if len(x) < block_size + 1:
            x = torch.cat([x, torch.zeros(block_size + 1 - len(x), dtype=torch.long)])
        
        return x

    
  def __len__(self):
     return max(1, len(self.data) - block_size)
from torch.utils.data import DataLoader
dataset=MultiLabelDataset(data)
def collate_fn(batch):
    batch = torch.stack(batch)  # Batch içindeki tüm tensörleri yığ
    data = batch[:, :-1]  # Son eleman hariç hepsi giriş verisi
    target = batch[:, 1:]  # 1 kaydırılmış veri hedef olarak kullanılıyor
    return data, target

train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=True)
"""

# data loading
def get_batch(split):
    # generate a sequential batch of data of inputs x and targets y

    # Veriyi sıralı almak için sabit bir başlangıç noktası belirleyelim
    global current_index
    if current_index + batch_size >= len(data_en) - block_size:
        current_index = 0  # Eğer veri biterse başa dön

    ix = torch.arange(current_index, current_index + batch_size)
    x = torch.stack([data_en[i:i+block_size] for i in ix])
    y = torch.stack([data_en[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    current_index += batch_size  # Bir sonraki batch için index'i güncelle
    
    return x, y
"""def get_batch(split):
    global current_index
    
    # Veri bitince başa sar
    if current_index + batch_size >= len(data) - block_size:
        current_index = 0  
    
    # Batch için verileri seç
    ix = torch.arange(current_index, current_index + batch_size)
    x_raw = [data[i:i+block_size].tolist() for i in ix]
    y_raw = [data[i+1:i+block_size+1].tolist() for i in ix]

    # Tensor’a çevirme
    x = torch.tensor(x_raw, dtype=torch.long, device=device)
    y = torch.tensor(y_raw, dtype=torch.long, device=device)

    current_index += batch_size  # Batch indeksini güncelle
    
    return x, y"""

# Başlangıç indexi
current_index = 0

"""class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y"""

class AltanMultiHead(nn.Module):
    def __init__(self,embed_dim,num_head):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_head, dropout=dropout)
    def forward(self, x):
        B, T, C = x.shape  # B = batch_size, T = sequence_length, C = embed_dim

        # Burada query, key, value aynı girdi verisinden alınır
        query = key = value = x.permute(1, 0, 2)  # (sequence_length, batch_size, embed_dim) şeklinde permütasyon yapıyoruz
        
        # Multihead attention hesaplama
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        
       
        
        # Çıkış
        return attn_output.permute(1, 0, 2)  # Çıkışı (B, T, C) şeklinde geri dönüyoruz



import torch
import numpy as np

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout=nn.Dropout(0.2)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
        return x

    
class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd,n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd,n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout=nn.Dropout(0.2)
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y=self.dropout(y)
        return y
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CausalSelfAttention(n_embd,n_head)
        self.ffwd = FeedFoward(n_embd)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
       
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        

        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(6)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        #self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        

        # weight sharing scheme
        #self.token_embedding_table.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B,T=idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)7
        for block_n in self.blocks:
            x = block_n(x) # (B,T,C)
        logits = self.ln_f(x) # (B,T,C)
        #logits = self.lm_head(x) 
        # (B,T,vocab_size)
        
        if targets is None:
            loss = None
            
            return logits
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
            return logits, loss
        """# forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)"""
       
        return logits

m=Encoder().to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')




"""rain_data_encode=train_dataloader
iter_encode=iter(train_data_encode)"""
"""for minwi_batch in range(2):
    for mini_batch in range(4):
            
            
            data_en,targets_en=next(iter_encode)
            
            print(data_en.shape)
            print(m(data_en.to("cuda")))"""
#print(find_encode(m,["First Citizen <EOS"">"]))
"""class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size):
        super().__init__()
        B,T,C=n_embd/n_head,block_size,n_embd
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = self.position_embedding_table(torch.arange(T, device=device)).unsqueeze(0).expand(B, T, -1)  # (B, T, C)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(6)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def forward(self, x):
        B, T = x.shape
        token_embeddings = self.embed(x)  # Token embedding
        position_embeddings = self.pos_embed[:, :T, :]  # Positional encoding
        x = token_embeddings + position_embeddings  # Embeddingler toplanır
        x = self.blocks(x)  # Transformer bloklarından geçir
        x = self.ln(x)  # Son LayerNorm
        return x
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
         # (B,T,vocab_size)

        return x

m=Encoder(vocab_size,n_embd,n_head,block_size).to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
xb, yb = get_batch('train')
output=m(xb).detach()
print(output.shape)"""