
from encoder import m as model_encode,data_en as data_encode
import sub_docs.config as config
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import inspect
import pandas as pd
import numpy as np
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size =64  # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = 16
n_layer = 4
dropout = 0.0
# ------------





# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


"""with open("sources/decoder_datas_for_ai.txt", 'r', encoding='utf-8') as f:
    text = f.read()
strings = text.split("\n")"""

strings=pd.read_csv("sources/translator_source_1.csv")
strings=list(np.array(strings["tr"]))[:50000]
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
data = torch.tensor(padsequences, dtype=torch.long).flatten()


# data loading
def get_batch(split):
    # generate a sequential batch of data of inputs x and targets y

    # Veriyi sıralı almak için sabit bir başlangıç noktası belirleyelim
    global current_index
    if current_index + batch_size >= len(data) - block_size:
        current_index = 0  # Eğer veri biterse başa dön

    ix = torch.arange(current_index, current_index + batch_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    current_index += batch_size  # Bir sonraki batch için index'i güncelle
    
    return x, y

# data loading
def get_batch_encode(split):
    # generate a sequential batch of data of inputs x and targets y

    # Veriyi sıralı almak için sabit bir başlangıç noktası belirleyelim
    global current_index_en
    if current_index_en + batch_size >= len(data_encode) - block_size:
        current_index_en = 0  # Eğer veri biterse başa dön

    ix = torch.arange(current_index_en, current_index_en + batch_size)
    x = torch.stack([data_encode[i:i+block_size] for i in ix])
    y = torch.stack([data_encode[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    current_index_en += batch_size  # Bir sonraki batch için index'i güncelle
    
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

    # Pad işlemi
    x_padded = pad_sequences(x_raw, maxlen=block_size, padding='post', value=0)
    y_padded = pad_sequences(y_raw, maxlen=block_size, padding='post', value=0)

    # Tensor’a çevirme
    x = torch.tensor(x_padded, dtype=torch.long, device=device)
    y = torch.tensor(y_padded, dtype=torch.long, device=device)

    current_index += batch_size  # Batch indeksini güncelle
    
    return x, y"""
# Başlangıç indexi
current_index = 0
current_index_en=0
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_en_e,y_en_e=get_batch_encode(split)
            X, Y = get_batch(split)
            output_x=model_encode(x_en_e)
            logits, loss = model(X,output_x, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
        y = self.dropout(y)
        #print(f"this is for self : {y.shape}")
        return y


class CausalCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head  # Her başın boyutu

        # Query, Key ve Value için linear katmanlar
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)

        # Çıkış projeksiyonu
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropout katmanı
        self.dropout=nn.Dropout(0.2)

    def forward(self, x, x_2):
       
        B, T, C = x.size()     # Decoder input (target sequence)
        B, S, C=x_2.size()

        # Query, Key ve Value hesapla
        q = self.q_proj(x)    # (B, T, C) -> Decoder Query
        k = self.k_proj(x_2)  # (B, S, C) -> Encoder Key
        v = self.v_proj(x_2)  # (B, S, C) -> Encoder Value

        # Başlara böl ve transpoz yap
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, S, hs)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, S, hs)
        #print(f"this is for cross values {q.shape} and {k.shape} and {v.shape}")
        # Scaled dot-product attention (Cross-Attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B, nh, T, hs)

        # Başları birleştir
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        #print(f"this is for cross y : {y.shape}")
        # Çıkış projeksiyonu ve dropout
        y = self.c_proj(y)
        y = self.dropout(y)
        
        return y
        
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
       
        
        return x
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CausalSelfAttention(n_embd,n_head)
        self.ffwd = FeedFoward(n_embd)
        self.cratt = CausalCrossAttention(n_embd,n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x,x2):
        x = x + self.sa(self.ln1(x))
        x = x + self.cratt(self.ln3(x),x2)
        x = x + self.ffwd(self.ln2(x))
        return x
    
class AltanTranslator(nn.Module):

    def __init__(self):
        super().__init__()
        

        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(6)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        

        # weight sharing scheme
        self.token_embedding_table.weight = self.lm_head.weight

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

    def forward(self, idx,x2, targets=None):
        # idx is of shape (B, T)
        B,T=idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, x2) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) 
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
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer     
    
    





model = AltanTranslator()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters for decoder')
import math
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps =6000  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
# create a PyTorch optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)


grad_accum_steps=16
num_steps = len(data_encode) // batch_size
print(len(data))
import sys;sys.exit()
for step in range(6250*2):#for 2 epoch
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

      # Toplam iterasyon sayısını hesapla
    for micro_step in range(grad_accum_steps):
        x_en, y_en = get_batch_encode("train")
        xb, yb = get_batch("train")
        
        x_en, y_en, xb, yb = (
            x_en.to(device),
            y_en.to(device),
            xb.to(device),
            yb.to(device),
        )

        output = model_encode(x_en)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, output, yb)

          # Gradient accumulation için loss ölçeklemesi
        loss=loss/grad_accum_steps

        loss.backward()

        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Learning rate ayarı
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()  # GPU işlemlerini tamamlamasını bekle
        loss_accum += loss.detach()    
        if(micro_step%5==False):
            print(f"Epoch {step+1} | Loss: {loss_accum.item():.6f} | LR: {lr:.4e} | Grad Norm: {norm:.4f}")










"""for iter in range(max_iters):

    
        
        

    # sample a batch of data
    x_en, y_en = get_batch_encode("train")
    xb, yb = get_batch('train')
    output=model_encode(x_en)
    # evaluate the loss
    logits, loss = m(xb,output, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"step {iter}: train loss {loss:.4f}")
    torch.cuda.synchronize()"""
    

    
