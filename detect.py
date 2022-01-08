"""
Detect whether or not mains noise is present in an audio signal 
and if its present, whether its present for the whole recording 
or appears and reappears.
"""

from cli import args

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import iirpeak, filtfilt, resample
from scipy.fft import rfft, next_fast_len, rfftfreq

# if you are using your own data, make sure its mono
fs_orig, data_orig = wavfile.read(args.input)
fs = 500
data = resample(data_orig, int((data_orig.shape[0] / fs_orig) * fs))

# filter frequencies that are not close to 50Hz
b, a = iirpeak(args.enf, 30, fs=fs)
data = filtfilt(b, a, data)

# frequency analysis
window_size = 5000
pad_to = next_fast_len(window_size, real=True)
window = data[:window_size] * np.hanning(window_size)
fourier = np.abs(rfft(window, n=pad_to))
frequencies = rfftfreq(window_size, d=1/fs)

# plot results
up_to = int(60 / (fs/window_size))
plt.plot(frequencies[:up_to], fourier[:up_to])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amount")
plt.title("Discrete time Fourier transform")
plt.show()
print("Main frequency at", np.argmax(fourier) * fs/window_size, "Hz")
