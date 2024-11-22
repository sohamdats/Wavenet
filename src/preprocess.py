import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

import math
import os
import glob

# hyperparameters
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--chunk_size", type=int, default=16000)
parser.add_argument("--hop_size", type=int, default=8000)
parser.add_argument("--split", type=float, default=0.8)

args = parser.parse_args()

class MusicDataset(Dataset):
    
    def __init__(self, filepath, train=True, transform=False):
        
        genre_files = {}
        for genere_fol in os.listdir(filepath):
            genre_path = os.path.join(filepath, genere_fol)
            genre_files[genere_fol] = glob.glob(os.path.join(genre_path, "*.mp3"))

        
        self.chunks = []
        genreid = {}
        
        for genre_idx, (genre, files) in enumerate(genre_files.items()):
            
            genreid[genre_idx] = genre
            
            self.transform = transform
            self.mu_law = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
            
            chunks = []
            chunk_size = args.chunk_size
            hop_size = args.hop_size
            
            cut_off = int(len(files) * args.split)
            
            if train:    
                files = files[:cut_off]
            else:
                files = files[cut_off:]
            
            for file in files:

                waveform, _  = torchaudio.load(file)

                if waveform.shape[0] == 2:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if self.transform:
                    waveform = self.mu_law(waveform)
                    
                num_chunks = int(math.ceil(waveform.shape[1]  / hop_size))
                
                for chunk_idx in range(0, num_chunks):
                    
                    start_idx = chunk_idx * hop_size
                    end_idx = min(waveform.shape[1], start_idx + chunk_size)
                    chunk = waveform[:, start_idx: end_idx]
                    
                    if chunk.shape[1] < chunk_size:
                        padding_size = chunk_size - chunk.shape[1]
                        last_element = chunk[:, -1:]
                        padding = last_element.repeat(1, padding_size)
                        chunk = torch.cat([chunk, padding], dim=1)
                        
                    chunks.append(chunk)
                    
            self.chunks.extend([(chunk, torch.tensor([[genre_idx]], dtype=torch.long)) for chunk in chunks])
            
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]
   
    