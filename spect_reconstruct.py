import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image
import seaborn as sns

import torch
import torchaudio

import random
from glob import glob

import librosa
import librosa.display
#import ffprobe
#import ffmpeg
import stempeg
import IPython.display as ipd

import musdb
import scipy
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftshift

from itertools import cycle
from scipy.io.wavfile import write

from ipywidgets import interact, interactive, fixed, interact_manual

import os

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])