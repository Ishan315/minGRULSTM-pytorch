import time
import math
import gzip
import random
import tqdm
import numpy as np
import wandb  # Import WandB

import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from minGRU_pytorch.minGRULM import minGRULM

# constants

NUM_BATCHES = int(3e3) + 5
BATCH_SIZE = 8
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 3e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 250
GENERATE_LENGTH = 512
SEQ_LEN = 512
NUM_TOKENS = 256
DIM = 512
DEPTH = 6
FF_LSTM = True

# Initialize WandB
wandb.init(
    project="minGRULSTM-LMm",
    config={
        "num_batches": NUM_BATCHES,
        "batch_size": BATCH_SIZE,
        "grad_accum_every": GRAD_ACCUM_EVERY,
        "learning_rate": LEARNING_RATE,
        "validate_every": VALIDATE_EVERY,
        "prime_length": PRIME_LENGTH,
        "generate_every": GENERATE_EVERY,
        "generate_length": GENERATE_LENGTH,
        "seq_len": SEQ_LEN,
        "model": "minGRULM",
        "num_tokens": NUM_TOKENS,
        "dim": DIM,
        "depth": DEPTH,
        "ff_lstm": FF_LSTM,
    },
    name=f"run_{int(time.time())}",  # Optional: name your run
    reinit=True
)

config = wandb.config  # Access the config if needed

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    prev_hiddens = None

    for _ in range(sample_num_times):
        logits, prev_hiddens = net(out, return_prev_hiddens = True)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# the minGRU char language model

model = minGRULM(
    num_tokens = NUM_TOKENS,
    dim = DIM,
    depth = DEPTH,
    ff_lstm = FF_LSTM 
).cuda()

# specs of the model
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# Watch the model with WandB (optional)
wandb.watch(model, log="all")

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# Training loop

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = model(data, return_loss = True)

        (loss / GRAD_ACCUM_EVERY).backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Step optimizer and zero gradients
    optim.step()
    optim.zero_grad()

    # Log training loss and learning rate
    current_lr = optim.param_groups[0]['lr']
    wandb.log({
        "train_loss": loss.item(),
        "learning_rate": current_lr,
    }, step=i, commit=False)

    # Print training loss
    print(f"Batch {i}: training loss: {loss.item():.3f}")

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            val_loss = model(valid_data, return_loss = True)
            print(f"Batch {i}: validation loss: {val_loss.item():.3f}")

            # Log validation loss
            wandb.log({
                "validation_loss": val_loss.item(),
            }, step=i, commit=True)

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        prime = decode_tokens(inp)
        print(f"%s \n\n %s" % (prime, "*" * 100))  # Fixed the print statement

        prompt = inp[None, ...]

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print("\n\n", base_decode_output, "\n")

# Save the model checkpoint and log it to WandB
checkpoint_path = f'checkpoint_{int(time.time())}.pt'
torch.save(model.state_dict(), checkpoint_path)
wandb.save(checkpoint_path)

# Optionally, finish the WandB run
wandb.finish()

