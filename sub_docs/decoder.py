from encoder import m as model_encode,get_batch as gb_encode,data_en as data_encode
import inspect
  # (1, T, C) formatında olmalı
import sub_docs.config as config
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
from torch.utils.data import Dataset
import gc
from accelerate import Accelerator

accelerator = Accelerator()
# Tüm GPU tensorlerini temizle

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 100
learning_rate = 1e-4
if torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    cuda_version = torch.version.cuda

    print(f"CUDA destekleniyor! Kullanılan cihaz: {device_name} ({device})")
    print(f"Toplam GPU sayısı: {device_count}")
    print(f"CUDA sürümü: {cuda_version}")

eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 6
dropout = 0.0
# ------------


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


with open(r"C:\Users\bahaa\Downloads\Birinci Yurttaş_eos.txt", 'r', encoding='utf-8') as f:
    text = f.read()
strings = text.split("\n")

# Tokenizer işlemi
tokenizer = Tokenizer()
tokenizer.fit_on_texts(strings)
stoi = tokenizer.word_index
itos = dict(zip(stoi.values(), stoi.keys()))
vocab_size = len(stoi)

# Metni sayılara çevirme ve padding
sequences = tokenizer.texts_to_sequences(strings)
padsequences = pad_sequences(sequences, maxlen=block_size, padding='pre')

# Tensor formatına çevirme (flatten ile tek boyuta indirgeme)
data = torch.tensor(padsequences, dtype=torch.long).flatten()





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



train_dataloader_decode = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=True)"""

# data loading
def get_batch_de(split):
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
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
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

class CrossAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    """def forward(self, x_1, x_2=output):
        B,T,C=x_1.shape
        queries_1 = self.query(x_1)
        keys_2 = self.key(x_2)
        values_2 = self.value(x_2)
        
        attn_scores = queries_1 @ keys_2.transpose(-2, -1)
        attn_weights = F.softmax(attn_scores / queries_1.size(-1) ** 0.5, dim=-1)
        context_vec = attn_weights @ values_2
        return context_vec"""
    def forward(self, x_1, x_2):
        # x_1: (batch_size, target_seq_len, embed_dim)
        # x_2: (batch_size, source_seq_len, embed_dim)
        B, T1, C = x_1.shape
        _, T2, _ = x_2.shape

        # Linearly project input tensors to query, key, and value
        queries = self.query(x_1)  # (B, T1, head_size)
        keys = self.key(x_2)      # (B, T2, head_size)
        values = self.value(x_2)  # (B, T2, head_size)

        # Compute attention scores (scaled dot-product attention)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, T1, T2)
        attn_scores = attn_scores / (C ** 0.5)  # Scale by the square root of head_size

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T1, T2)
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors
        context_vec = torch.matmul(attn_weights, values)  # (B, T1, head_size)

        return context_vec
        
"""class MultiCrossAttention(nn.Module):
    

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out"""
"""class MultiCrossAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by num_heads"
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([CrossAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)  # Combine all heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1, x_2):
        # Multi-Head Cross Attention
        out = torch.cat([head(x_1, x_2) for head in self.heads], dim=-1)  # (B, T, embed_dim)
        out = self.dropout(self.proj(out))  # Final projection
        return out"""


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
class CausalCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head  # Her bir başın boyutu

        # Query, Key ve Value için ayrı linear katmanlar
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)

        # Çıkış projeksiyonu
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Düzenleme katmanı
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x_2):
        B, T, C = x.size()  # x = (B, T, C), decoder input (target sequence)
        B2, S, C2 = x_2.size()  # x_2 = (B, S, C), encoder output (source sequence)

        # Query, Key ve Value hesapla
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x_2)  # (B, S, C)
        v = self.v_proj(x_2)  # (B, S, C)

        # Başlara böl ve transpoz yap
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, S, hs)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, S, hs)

        # Scaled dot-product attention (Cross-Attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B, nh, T, hs)

        # Başları birleştir
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Çıkış projeksiyonu
        y = self.c_proj(y)
        y = self.dropout(y)
        
        return y
    
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
        self.cratt = CausalCrossAttention(n_embd,n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x,x2):
        x = x + self.sa(self.ln1(x))
        x = x + self.cratt(self.ln3(x),x2)
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
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
    
total_batch=batch_size*block_size*4
accu=4
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 128
max_steps=2000
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



model = AltanTranslator()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# create a PyTorch optimizer
#optimizer = m.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
optimizer = torch.optim.AdamW(m.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)

"""class Data_Collactor(nn.Module):
    def __init__(self,de,en):
        super().__init__()
        self.en=en
        self.de=de
    def __call__(self):
        en_list=[]
        de_list=[]
        for idx, (data_en, targets_en) in enumerate(tqdm(self.en)):
            en_list.append(data_en)

        for idx, (data_de, targets_de) in enumerate(tqdm(self.de)):
            
            de_list.append((data_de,targets_de))
        return en_list,de_list
data_collactor=Data_Collactor(train_dataloader_decode,train_data_encode)    
en_list,de_list=data_collactor()
print(en_list[0].shape)
a,b=de_list[0]
print(a.shape)"""

"""iter_encode=iter(train_data_encode)
iter_Decode=iter(train_dataloader_decode)
"""

import time

      # Sıfırlama burada yapılabilir

accumulation_steps = 4 # 4 batch biriktir, sonra güncelle
for epoch in range(3000):
    loss_Accum = 0
    optimizer.zero_grad()
    for batch_idx in range(1):
        x_en, y_en = gb_encode("train")
        x_de,y_de=get_batch_de("train")
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                  # Batch al
            output=model_encode(x_en)
            
            logits, loss = m(x_de,output,y_de)
        loss = loss / 4
        loss_Accum += loss.detach()
        accelerator.backward(loss)

        # Sadece belirli adımlarda optimizer.step() çağır
        scheduler.step(loss_Accum.item())
        if (batch_idx + 1) % accu == 0:
            optimizer.step()
            optimizer.zero_grad()
    if(epoch%5==0):
        print(f"step {epoch:5d} | loss: {loss_Accum.item():.6f} | lr {optimizer.param_groups[0]['lr']:.4e}")







"""for iter in range(max_steps):
    loss_Accum=0
    for idx, (data_en, targets_en) in enumerate(tqdm(train_data_encode)):
        data_encode, targets_encode = data_en.to(device), targets_en.to(device)
        
        outputs = model_encode(data_encode)
    optimizer.zero_grad(set_to_none=True)      # Extract pooled output for classification
    for idx, (data_de, targets_de) in enumerate(tqdm(train_dataloader_decode)):
        data_de, targets_de = data_de.to(device), targets_de.to(device)
        
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits, loss = model(data_de,targets_de,outputs)
          # Extract pooled output for classification
        loss=loss/accu
        loss_Accum+=loss.detach()
        
        loss.backward()
        lr=get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr
        if (idx + 1) % accu == 0 or (idx + 1) == len(train_dataloader_decode):
            optimizer.step()
            optimizer.zero_grad()
    """
    # sample a batch of data
    
    
    
        
