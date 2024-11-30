#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
class SampleGenerator(object):
    """Construct dataset for multi-class classification"""

    def __init__(self, ratings, n_test=1):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # self.preprocessed_ratings = self._preprocess_ratings(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.ratings, n_test)

    def _split_loo(self, ratings, n_test=1):
        """
        Split dataset into train and test sets, with `n_test` interactions per user in the test set.

        Args:
            ratings: DataFrame, contains user-item interactions.
            n_test: int, number of recent interactions to include in the test set.

        Returns:
            train: DataFrame, contains the train set.
            test: DataFrame, contains the test set.
        """
        # Rank interactions by timestamp for each user
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

        # Test set contains the top `n_test` interactions for each user
        test = ratings[ratings['rank_latest'] <= n_test]
        
        # Train set contains all other interactions
        train = ratings[ratings['rank_latest'] > n_test]

        # Ensure every user is in both train and test sets
        assert train['userId'].nunique() == test['userId'].nunique()

        # Return train and test dataframes
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def evaluate_data(self):
        """Create evaluate data for classification"""
        test_users = self.test_ratings['userId'].tolist()
        test_items = self.test_ratings['itemId'].tolist()
        test_ratings = self.test_ratings['rating'].tolist()

        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(test_ratings)]
    
    def get_train_test_dataframes(self):
        """
        Returns the train and test dataframes.
        """
        return self.train_ratings, self.test_ratings


# In[3]:


import pandas as pd
import numpy as np
from IPython.display import display

# Load Data
data_dir = '/raid/home/shyamsg/Final_Project/DecoderOnlyModel/Data/u1m.dat'
rating_df = pd.read_csv(data_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

# i want to drop nulls
rating_df = rating_df.dropna()

# Reindex userId and itemId, start from 0
user_id = rating_df[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
rating_df = pd.merge(rating_df, user_id, on=['uid'], how='left')

# Reindex itemId, start from 0
item_id = rating_df[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
rating_df = pd.merge(rating_df, item_id, on=['mid'], how='left')

rating_df = rating_df[['userId', 'itemId', 'rating', 'timestamp']]

print('Range of userId is [{}, {}]'.format(rating_df.userId.min(), rating_df.userId.max()))
print('Range of itemId is [{}, {}]'.format(rating_df.itemId.min(), rating_df.itemId.max()))


# n_test specifies number of most recent interactions to be used for testing for each user
sample_generator = SampleGenerator(rating_df, n_test=15)

# Get train and test dataframes
df1, df2 = sample_generator.get_train_test_dataframes()

print(df1.shape, df2.shape)


# In[4]:


# now create a new dataframe columns
#df['prompt'] = What star rating do you think userid_df[user_id] will give item_df[item_id]? 
#df['rating'] = df[rating]

train_df = pd.DataFrame()
train_df['prompt'] = "What star rating do you think user_" + df1['userId'].astype(str) + " will give item_" + df1['itemId'].astype(str) + "?"
train_df['rating'] = df1['rating']
print(train_df.shape)   

val_df = pd.DataFrame()
val_df['prompt'] = "What star rating do you think user_" + df2['userId'].astype(str) + " will give item_" + df2['itemId'].astype(str) + "?"
val_df['rating'] = df2['rating']
print(val_df.shape)

print(f"Train Size: {len(train_df)}, Val Size: {len(val_df)}")
          


# In[8]:


"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
Code adopted from: https://github.com/karpathy/nanoGPT
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification


################################################################################
# GPT Model classes
################################################################################

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention, but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(
                torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size))
        
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)


        attn_out = self.c_attn(x)

        # TODO: send x through the auxiliary c_attn and add them back to attn_out
        # x = self.c_attn_GPT2model(x)
        # attn_out = attn_out + x
        # print(f"shape of x in CausalSelfAttention: {attn_out.shape}")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = attn_out.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        proj_out = self.c_proj(y)

        # TODO: send y through the auxiliary c_proj and add them back to proj_out
        # added by me
        # proj_out = proj_out + self.c_proj_GPT2model(y)
        # added by me 

        y = self.resid_dropout(proj_out)
        
        # print(f"shape of y after residual dropout: {y.shape}")
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # added by me 

    def forward(self, x):
        # commented by me which they have given
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
    
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, model_type='gpt2', is_gen=False):
        super(GPT, self).__init__()
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-small':  dict(n_layer=12, n_head=12, n_embd=768), # later last 6 layers will be removed
            # 'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            # 'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            # 'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M (1.3B) params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True        # always True for GPT model checkpoints

        self.config = GPTConfig(**config_args)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))
        if is_gen:
            self.lm_head = nn.Linear(
                self.config.n_embd, self.config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
            self.score = None
        else:
            self.score = nn.Linear(self.config.n_embd, 5, bias=False)

        # TODO: Remove gradients from the embedding layers
        # added by me below
        self.transformer.wte.weight.requires_grad = False
        self.transformer.wpe.weight.requires_grad = False
        # added by me above 

        sd = self.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        print("loading weights from pretrained gpt2 model")
        if is_gen:
            model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            model_hf = GPT2ForSequenceClassification.from_pretrained('gpt2')

        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # intialize the score.weight with xavier initialization 
        for k in sd_keys:
            if k.endswith('score.weight'):
                nn.init.xavier_uniform_(sd[k])
        # remove the score.weigth from the list of keys
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('score.weight')]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters except the params that are not in the pretrained model
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        # Remove the last 6 layers for gpt2-small variant
        if model_type == 'gpt2-small':
            self.transformer.h = self.transformer.h[:6]
                    
        # print the total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params / 1e6:.2f}M")
        if not is_gen:
            # print the number of trainable parameters
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
            # calculate the reduction in parameters
            reduction = 100 * (total_params - num_params) / total_params
            print(f"Reduction: {reduction:.2f}%")

    def forward(self, idx, mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # if mask is provided, find the indices of the last tokens in each sequence
        if mask is not None:
            assert mask.size() == idx.size(), "Mask size must match input size"
            eos_idxs = mask.sum(1) - 1 # last non-pad token in each sequence

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if self.score is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        else: # no need to preserve the time dimension for classification task
            if mask is not None:
                # if mask is provided, only return the logits for the last token in each sequence
                logits = self.score(x[torch.arange(b, device=device), eos_idxs])
            else:
                
                logits = self.score(x[:, -1, :])
               
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
       
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            
            # print(f"shape of idx in generate: {idx.shape}") # added by me 
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def save_trainable_params(self, path):
        trainable_params =\
            list(filter(lambda p: p.requires_grad, self.parameters()))
        torch.save(trainable_params, path)
    
    def load_trainable_params(self, path):
        trainable_params = torch.load(path)
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = trainable_params.pop(0)


# In[9]:


def get_data_loader(data, batch_size, tokenizer, shuffle, max_len=40):
    """
    Get a data loader for the training data.
    """
    X, y = data['prompt'], data['rating']
    y= y-1
    # print(X)
    # print("type of X: ", type(X))
    # print("type of y: ", type(y))
    X = tokenizer.batch_encode_plus(
        X.tolist(), max_length=max_len, truncation=True, padding='max_length')
    X, mask = X['input_ids'], X['attention_mask']
    # convert them to tensors
    X = torch.tensor(X)
    mask = torch.tensor(mask)
    y = torch.tensor(y.values, dtype=int)
    data = torch.utils.data.TensorDataset(X, mask, y)
    
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader


# In[10]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# In[ ]:


import torch
import time
import matplotlib.pyplot as plt

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_pred = 0
    total_pred = 0
    for batch in train_loader:
        optimizer.zero_grad()
        X, mask, y = batch
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        output = model(X, mask) # return logits for last valid token
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct_pred += (predicted == y).sum().item()
        total_pred += len(y)
    
    return total_loss / len(train_loader), correct_pred / total_pred 
  
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_pred = 0
    total_pred = 0
    with torch.no_grad():
        for batch in val_loader:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            output = model(X, mask)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_pred += (predicted == y).sum().item()
            total_pred += len(y)
             
    return total_loss / len(val_loader) , correct_pred / total_pred 

    
def plot_losses(train_losses, val_losses, mode, args):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{mode}_{args.gpt_variant} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'../plots_1m/{args.gpt_variant}_loss.png')
    plt.close()
    print(f"Plots saved at plots_1m/{args.gpt_variant}_loss.png")
    
def plot_metrics(train_accs, val_accs, mode, args):
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{mode} Accuracy')
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'../plots_1m/{args.gpt_variant}_acc.png')
    plt.close()
    print(f"Plots saved at plots_1m/{args.gpt_variant}_acc.png")


# In[12]:


train_loader = get_data_loader(train_df, 256, tokenizer,True)
val_loader = get_data_loader(val_df, 256, tokenizer,False)

print("Length of train_loader: ", len(train_loader))
print("Length of val_loader: ", len(val_loader))


import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
import argparse



# Define the main function and other necessary functions
def main(args):
    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die. "
        # prompt = "Once upon a time, in a land far far away, there was a"

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "GPT2model":    
        model = GPT(args.gpt_variant).to(args.device)
        
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(args.epochs):
            print('-' * 80)
            epoch_start = time.time()

            train_start= time.time()
            train_loss,train_acc = train(model, train_loader, optimizer, criterion, args.device)
            train_end= time.time()

            val_start= time.time()
            val_loss,val_acc = evaluate(model, val_loader, criterion, args.device)
            val_end= time.time()

            epoch_end = time.time()

            # Calculate the durations
            train_duration = train_end - train_start
            val_duration = val_end - val_start
            epoch_duration = epoch_end - epoch_start
    
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Time taken for epoch {epoch+1}: {epoch_duration:.4f} seconds (Training: {train_duration:.4f} seconds, Evaluation: {val_duration:.4f} seconds)")
            # save the model after every 10 epochs in model directory with different names use save_trainable_params
            # if (epoch + 1) % 25 == 0:
            #     model.save_trainable_params(f'model/{args.gpt_variant}_{epoch+1}.pth')

        with open(f'../results_1m/{args.gpt_variant}_metrics.txt', 'w') as f:
            f.write('Train Losses:\n')
            f.write(str(train_losses))
            f.write('\nVal Losses:\n')
            f.write(str(val_losses))
            f.write('\nTrain Accuracies:\n')
            f.write(str(train_accs))
            f.write('\nVal Accuracies:\n')
        print(f"Metrics saved at model/{args.gpt_variant}_metrics.txt") 
        
        # TODO: Also plot the training losses and metrics
        # save the plot in plots folder 
        plot_losses(train_losses, val_losses, args.mode, args)
        plot_metrics(train_accs, val_accs, args.mode, args)
        # model.save_trainable_params(args.model_path)
    else:
        print("Invalid mode")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT2 Model Training and Generation")
    parser.add_argument("--mode", type=str, default="GPT2model", help="Mode of operation: gen or GPT2model")
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2-small", help="GPT variant to use: gpt2, gpt2-small")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/GPT2model.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")

    args = parser.parse_args()
    args.device = torch.device(
        "cuda:2" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")

    main(args)

# In[ ]:




