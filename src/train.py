import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import sys

from preprocess import MusicDataset
from model import Wavenet

import argparse
import wandb

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true")

parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--n_embeddings", type=int, default=256,
    help='number of embeddings')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

parser.add_argument("--project_name", type=str, default="wavenet-training",
    help='wandb project name')
parser.add_argument("--run_name", type=str, default=None,
    help='wandb run name')

args = parser.parse_args()

wandb.init(
    project=args.project_name,
    name=args.run_name,
    config=vars(args)
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if "cuda" in device:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

## Loading dataset
dataset_path = os.path.join("..", "data", "genrenew")
train_music_dataset = MusicDataset(dataset_path, transform=True)
valid_music_dataset = MusicDataset(dataset_path, train=False, transform=True)

train_loader = DataLoader(train_music_dataset, 
                          batch_size=args.batch_size, 
                          shuffle=True,
                          num_workers=args.num_workers, 
                          pin_memory=True)

test_loader = DataLoader(valid_music_dataset, 
                          batch_size=args.batch_size, 
                          shuffle=True,
                          num_workers=args.num_workers, 
                          pin_memory=True)

## Loading model
model = Wavenet(input_dim=args.n_embeddings, num_layers=args.n_layers)
model.to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

## train , test and log

def train():
    
    start_time = time.time()
  
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        
        x = x.to(device)
        label = label.to(device)
        
        output = model(x, label)
        logits = output.permute(0, 2, 3, 1).contiguous()
        
        loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_loss.append(loss.item())
        
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = np.asarray(train_loss)[-args.log_interval:].mean(0)
            wandb.log({
                "train_loss": avg_loss,
                "train_step": batch_idx * len(x)
            })
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                args.log_interval * batch_idx / len(train_loader),
                avg_loss,
                time.time() - start_time
            ))



def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x = x.to(device)
            label = label.to(device)
            
            output = model(x, label)
            logits = output.permute(0, 2, 3, 1).contiguous()
                    
            loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            val_loss.append(loss.item())

    val_loss = np.asarray(val_loss).mean(0)
    wandb.log({
        "val_loss": val_loss,
        "epoch": epoch
    })
    print('Validation Completed!\tLoss: {} Time: {}'.format(
        val_loss,
        time.time() - start_time
    ))
    return val_loss



BEST_LOSS = 999
LAST_SAVED = -1

os.makedirs('results', exist_ok=True)
print("Training started")
for epoch in range(1, args.epochs):
    print("\nEpoch {}:".format(epoch))
    train()
    cur_loss = test()

    if args.save or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), 'results/{}_wavenet.pt'.format(args.dataset))
        wandb.log({
            "best_val_loss": BEST_LOSS,
            "last_saved_epoch": LAST_SAVED
        })
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
   
wandb.finish()
   