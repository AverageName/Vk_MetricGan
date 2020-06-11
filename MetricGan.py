import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from networks import LstmGen, ConvDiscriminator
from utils import *

class MetricGan(nn.Module):


    def __init__(self, input_features, hidden_dim, num_layers, fc_hidden_dim, metric, device):
        super(MetricGan, self).__init__()
        self.G = LstmGen(input_features, hidden_dim, num_layers, fc_hidden_dim).to(device)
        self.D = ConvDiscriminator().to(device)
        self.metric = metric
        self.device = device
    
    def forward(self, batch):
        noisy = batch["noisy"].to(self.device)
        noisy_wo_norm = batch["noisy_wo_norm"].to(self.device)

        mask = torch.clamp(self.G(noisy), min=0.05)
        denoised = noisy_wo_norm * mask

        return denoised
    
    def backward_G(self, batch):
        clean_wo_norm = batch["clean_wo_norm"].to(self.device)

        denoised = self.forward(batch)
        logits = self.D(torch.stack([denoised, clean_wo_norm], dim=1))
        #print(logits)
        #Magical number 1
        loss = torch.sum((logits - 1.0)**2)
        loss.backward()
    
        return loss.item()

    def backward_D(self, batch):
        clean_wo_norm = batch["clean_wo_norm"].to(self.device)

        denoised = self.forward(batch).detach()

        logits_noise = self.D(torch.stack([denoised, clean_wo_norm], dim=1))
        logits_clean = self.D(torch.stack([clean_wo_norm, clean_wo_norm], dim=1))

        denoised = np.multiply(denoised.squeeze(0).cpu().numpy().transpose(), np.exp(1j*batch['phase'].numpy()))
        denoised_1d = librosa.istft(np.squeeze(denoised, 0), hop_length=256,
                                    win_length=512, window=scipy.signal.hamming, length=batch['clean_array'].shape[1])
        

        clean_1d = batch['clean_array'].squeeze(0).numpy()

        metrics_noise = Q(self.metric, clean_1d, denoised_1d, batch['sample_rate'])
        
        if self.metric == "pesq":
            metrics_clean = Q(self.metric, clean_1d, clean_1d, batch['sample_rate'])
        else:
            metrics_clean = 1

        loss = (logits_noise - metrics_noise)**2 + (logits_clean - metrics_clean)**2
        loss.backward()

        return loss.item()