import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
from dataclasses import dataclass
import math
import os
import glob


def mu_law_encode(x, quantization_channels=256):
    """
    Encode audio signal using mu-law companding.
    
    Args:
        x (torch.Tensor): Input audio signal (-1 <= x <= 1)
        quantization_channels (int): Number of channels (default: 256)
    
    Returns:
        torch.Tensor: Encoded signal (0 <= output <= quantization_channels - 1)
    """
    mu = quantization_channels - 1
    
    # Clamp the input to [-1, 1]
    x = torch.clamp(x, -1.0, 1.0)
    
    # Compute mu-law companding
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu))
    
    # Scale to [0, quantization_channels-1]
    x_mu = ((x_mu + 1) / 2 * mu).long()
    
    return x_mu

def mu_law_decode(x_mu, quantization_channels=256):
    """
    Decode mu-law encoded signal back to audio.
    
    Args:
        x_mu (torch.Tensor): Encoded signal (0 <= x_mu <= quantization_channels - 1)
        quantization_channels (int): Number of channels (default: 256)
    
    Returns:
        torch.Tensor: Decoded audio signal (-1 <= output <= 1)
    """
    mu = quantization_channels - 1
    
    # Convert back to [-1, 1] range
    x_mu = x_mu.float()
    x = (2 * x_mu / mu) - 1
    
    # Invert mu-law companding
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(torch.tensor(mu))) - 1) / mu
    
    return x


# hyperparameters
@dataclass
class Config:
    chunk_size: int = 16000
    hop_size: int = 8000
    split: float = 0.8

# Use this config instance throughout your code
config = Config()

class MusicDataset(Dataset):
    
    def __init__(self, filepath, train=True, transform=True):
        
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
            chunk_size = config.chunk_size
            hop_size = config.hop_size
            
            cut_off = int(len(files) * config.split)
            
            if train:    
                files = files[:cut_off]
            else:
                files = files[cut_off:]
            
            for file in files:

                waveform, _  = torchaudio.load(file, normalize=True)

                if waveform.shape[0] == 2:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if self.transform:
                    # waveform = self.mu_law(waveform)  # to set the range [0, 255]
                   waveform = mu_law_encode(waveform)
                   
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
                    
            self.chunks.extend([(chunk, genre_idx) for chunk in chunks])
            
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]
   
    
    
if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=config.chunk_size)
    parser.add_argument("--hop_size", type=int, default=config.hop_size)
    parser.add_argument("--split", type=float, default=config.split)
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.chunk_size = args.chunk_size
    config.hop_size = args.hop_size
    config.split = args.split