"""
Detect mains noise in a video and
timestamp it, if possible.
"""

from cli import args

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import iirpeak, filtfilt, resample
from scipy.fft import rfft, next_fast_len

print(args)

# if you are using your own data, make sure its mono
fs_orig, data_orig = wavfile.read(args.input)
fs = 500
data = resample(data_orig, int((data_orig.shape[0] / fs_orig) * fs))

# filter frequencies that are not close to 50Hz
b, a = iirpeak(args.enf, 30, fs=fs)
data = filtfilt(b, a, data)

duration = 5 # length of the window in seconds
window_size = fs * duration # number of samples per window
pad_to = next_fast_len(window_size, real=True)

offset = fs # offset of each window
max_n_windows = 100 # maximum number of fft's to compute
n_windows = min(max_n_windows, (data.shape[0] - window_size)//offset) # total number of windows
enf_change = np.zeros(n_windows)

for i in range(n_windows):
    start = i * offset
    window = data[start:start + window_size] * np.hanning(window_size)
    fourier = np.abs(rfft(window, n=pad_to))
    enf_change[i] = np.argmax(fourier) * (fs/window_size)
    
plt.plot(enf_change)
plt.xlabel("Time (s)")
plt.ylabel("ENF")
plt.show()
