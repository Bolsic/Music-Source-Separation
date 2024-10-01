from model import MusicSep
import torch
import numpy
import os
from glob import glob
import torchaudio.transforms as transforms

def transform2to4(input_tensor):
    input_tensor = torch.stack((input_tensor[0, :, : ,0], input_tensor[0, :, : ,1], input_tensor[1, :, : ,0], input_tensor[1, :, : ,1]))
    return torch.stack((torch.complex(input_tensor[0], input_tensor[1]), torch.complex(input_tensor[2], input_tensor[3])))

def calculate(model, input_tensor):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.stack((input_tensor[0, :, : ,0], input_tensor[0, :, : ,1], input_tensor[1, :, : ,0], input_tensor[1, :, : ,1]))
        #print(input_tensor.size())
        input_tensor = input_tensor.to(dtype = torch.float32)
        output = model(input_tensor.unsqueeze(0))
        result = torch.multiply(output[0], input_tensor)
        #print("output[0] :", result.shape)

    return torch.stack((torch.complex(result[0], result[1]), torch.complex(result[2], result[3])))

def get_wav(spect_big, n_fft = 1024, hop_length = 512):
    transform = transforms.InverseSpectrogram(n_fft = n_fft, hop_length = hop_length)
    return transform(spect_big, int(15*16000))

