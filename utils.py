import glob
import os
import random
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import librosa
import scipy
import torch
from pesq import pesq
from pystoi import stoi
from torch_stoi import NegSTOILoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from scipy import signal
from scipy.io import wavfile


class Timit(Dataset):

    def __init__(self, path_to_clean, path_to_noisy):
        
        self.noisy_paths = glob.glob(os.path.join(path_to_noisy, '**/*.wav'), recursive=True)
        #self.num_noises = num_noises
        #self.every = num_noises
        self.path_to_noisy = path_to_noisy
        self.path_to_clean = path_to_clean
    
    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        
        # clean_audio_path = self.clean_paths[idx // self.every]
        # clean_wav_name = clean_audio_path.rsplit('/')[-1].strip('.WAV')
        # clean_folder = clean_audio_path.rsplit('/', 2)[-2]

        #noisy_folder = os.path.join(path_to_save, "{}_{}".format(clean_folder, wav_name))
        #folder_path = os.path.join(self.path_to_noisy, "{}_{}".format(clean_folder, clean_wav_name))
        noisy_audio_path = self.noisy_paths[idx]

        #print(noisy_audio_path)
        clean_audio_folder, clean_wav_name = noisy_audio_path.rsplit('/', 2)[-2].split('_')
        clean_audio_path = glob.glob(os.path.join(self.path_to_clean, '**', clean_audio_folder, clean_wav_name + ".WAV"))[0]
    
        #print(clean_audio_path)
        #print(noisy_audio_path)
        noisy_array, sample_rate = convert_audio_to_array(noisy_audio_path)
        spectr_noisy, phase_noisy, _ = get_spectr_and_phase(noisy_array, norm=True)
        spectr_noisy_wo_norm, _, _ = get_spectr_and_phase(noisy_array, norm=False)
        spectr_noisy_wo_norm = np.squeeze(spectr_noisy_wo_norm, 0)
        spectr_noisy = np.squeeze(spectr_noisy, 0)
        
        array, sample_rate = convert_audio_to_array(clean_audio_path)
        spectr, phase_train, _ = get_spectr_and_phase(array, norm=True)
        spectr_wo_norm, _, _ = get_spectr_and_phase(array, norm=False)
        spectr_wo_norm = np.squeeze(spectr_wo_norm, 0)
        spectr = np.squeeze(spectr, 0)


        return {'clean': spectr, 'noisy': spectr_noisy, 'noisy_wo_norm': spectr_noisy_wo_norm,
                'clean_wo_norm': spectr_wo_norm, 'phase': phase_noisy, 'clean_array': array, 'sample_rate': sample_rate}


def Q(mode, clean, denoised, sr):
    if mode == "pesq":
        #Figure out wb and nb what is it
        return pesq(sr, clean, denoised, 'wb')
    elif mode == "stoi":
        #criterion = NegSTOILoss(sample_rate=sr)
        #print(criterion(torch.from_numpy(denoised).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0)))
        #return stoi(clean, denoised, sr, extended=False)
        #return -torch.sum(criterion(torch.from_numpy(denoised).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0)))
        #clean = librosa.resample(clean, sr, 10000)
        #denoised = librosa.resample(denoised, sr, 10000)
        return stoi(clean, denoised, 16000, extended=False)


def generate_noisy_wavs(path_to_clean_wavs, clean_limit, snrs, path_to_noises, noise_types_wavs, path_to_save):
    clean_wavs = glob.glob(os.path.join(path_to_clean_wavs, '**/*.WAV'), recursive=True)[:clean_limit]
    #print(clean_wavs)
    for clean_wav_path in clean_wavs:
        wav_name = clean_wav_path.rsplit('/', 1)[-1].strip('.WAV')
        clean_folder = clean_wav_path.rsplit('/', 2)[-2]

        if not os.path.exists(os.path.join(path_to_save, "{}_{}".format(clean_folder, wav_name))):
            os.mkdir(os.path.join(path_to_save, "{}_{}".format(clean_folder, wav_name)))

        noisy_folder = os.path.join(path_to_save, "{}_{}".format(clean_folder, wav_name))
        clean_wav, sample_rate_orig = sf.read(clean_wav_path)
        rms_signal = np.mean(clean_wav**2)**(1/2)

        for snr in snrs:
            for noise_type_wav in noise_types_wavs:
                #noise_type_wav = random.choice(noise_types_wavs)
                noise_type_wav = os.path.join(path_to_noises, noise_type_wav)
                noise_name = noise_type_wav.rsplit('/', 1)[-1].strip('.wav')

                noise_wav, sample_rate = sf.read(noise_type_wav)
                rms_noise = np.mean(noise_wav**2)**(1/2)
                rms_noise_should = (rms_signal**2/(10**(snr/10)))**(1/2)
                noise = noise_wav * (rms_noise_should/rms_noise)

                noise_wav = clean_wav + np.concatenate([noise, noise, noise, noise, noise, noise, noise])[:clean_wav.shape[0]]

                sf.write(os.path.join(noisy_folder, 'snr_{}_noise_{}.wav'.format(str(snr), noise_name)), noise_wav, sample_rate_orig)


def convert_audio_to_array(wav_path):
    samples, sample_rate = sf.read(wav_path)
    return samples, sample_rate


def get_spectr_and_phase(signal, norm=False):
    #print(signal.shape)       
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    
    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    
    Lp=np.abs(F)
    phase=np.angle(F)
    if norm == True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257)) # For LSTM

    return NLp, phase, signal_length


def enchance_wav(model, wav_path):
    array, sr = convert_audio_to_array(wav_path)

    spectr_without_norm, phase, length = get_spectr_and_phase(array, False)
    spectr, _, _ = get_spectr_and_phase(array, True)

    inputs = torch.from_numpy(spectr).cuda()
    mask = model.G(inputs).detach().cpu().numpy()

    #mask = np.maximum(outputs / spectr, 0.05)
    outputs = np.squeeze(mask * spectr_without_norm)

    enchanced = np.multiply(outputs.transpose() , np.exp(1j*phase))

    denoised_1d = librosa.istft(enchanced, hop_length=256,
                                    win_length=512, window=scipy.signal.hamming, length=length)
    return array, denoised_1d, outputs, spectr_without_norm