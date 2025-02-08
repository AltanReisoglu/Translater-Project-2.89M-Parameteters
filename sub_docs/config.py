import torch
batch_size = 64
block_size = 128
max_iters = 50
eval_interval = 100
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.0