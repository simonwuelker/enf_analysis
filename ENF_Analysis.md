---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Electric Network Frequency Analysis
This Notebook is supposed to act as a step-by-step explanation of the
[Electric Network Frequency (ENF) Analysis script](https://github.com/Wuelle/enf_analysis) i made. 
I assume that the reader is familiar with the basic concept of a Fourier Transformation.
If you want to follow along, please run the downloader script and place an arbitrary audio sample in the `data/` directory.
Note that to keep this project fun, i tried to implement as much of the math as possible on my own.
Because of this, the notebook got quite large. Please make use of the table of contents below and skip around to the interesting parts.

<br>
<em>Supported by National Grid ESO Open Data<em>
<br>

## Table of Contents
* [Introduction](#Introduction)
* [Fourier Transformation](#Fourier-Transformation)
* [Short-time Fourier Transform](#Short-time-Fourier-Transform)

+++

## Introduction
The european powergrid runs on alternating current, with a frequency of, in theory, exactly 50 Hertz. In practice however,
this value fluctuates due to changes in supply and demand. These Fluctuations are usually in the order of < 1Hz and do not affect everyday consumers at all.

Another thing to realize is that alternating current can leave <abbr title="Electric Network Frequency">ENF</abbr>-Artifacts in audio recordings.
The noise is know as the [Mains Hum](https://en.wikipedia.org/wiki/Mains_hum) and in its pure 50 Hz form, it
is too deep to be heard by a human ear. 

Luckily, the british [National Grid ESO](https://data.nationalgrideso.com) provides a 
[free dataset](https://data.nationalgrideso.com/system/system-frequency-data?from=0#resources) of these frequency variations since January of 2019, which we can use to get the historical data. They record the frequency once per second, which is a lot more than i had initially hoped for.

If however, we extract this background noise and match it with the values from the dataset, we can exactly timestamp the recording.
This process has already been used in court to verify the authenticity of provided audio or to detect evidence of tampering.(because the <abbr title="Electric Network Frequency">ENF</abbr> would suddenly jump around)

Since i only have data from europe, my algorithm is only suitable for audio from europe. It should, however, be possible 
to adapt it to another frequency with only minimal changes. You can find your power grid frequency using the map below.

```{note}
Look at this cool map
![Powergrid frequencies across the world](https://upload.wikimedia.org/wikipedia/commons/7/70/World_Map_of_Mains_Voltages_and_Frequencies%2C_Detailed.svg)
*SomnusDe, Public domain, via Wikimedia Commons*
```

```{code-cell} ipython3
lol
```

But exactly by how much does the grid frequency fluctuate? According to the [National Grids Mandatory Frequency Response Guide](https://www.nationalgrid.com/sites/default/files/documents/Mandatory%20Frequency%20Response%20Guide%20v1.1.pdf), the grid is considered

|   | unoperational | unstatuory |
| --- | --- | --- |
| below | 49.8Hz | 49.5Hz |
| above | 50.2Hz | 50.5Hz |

We will therefore only perform fourier transformations for the frequencies between 49.5Hz and 50.5Hz. Since the dataset
has a precision up to $\frac{1}{100}$ of a Hertz, it would be good to match that. We therefore need to perform Fourier transformations for $(50.5 - 49.5) \times 100 = 100$ frequencies.
The Nyquist Frequency, which i explain [here](#Nyquist-Frequency), tells us that to analyze frequencies up to 50.5Hz, we will need (at least) a sample rate of 101 samples per second. This is great because .wav files usually sit at around 48000 samples per second. By reducing the number of samples per second, we can greatly reduce the computations required.
If we want precision up to $\frac{1}{100}$ and a maximum of 50.5Hz, we will require $50.5 \times 100 = 5050$ samples.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

%matplotlib inline
plt.rcParams["figure.figsize"] = (20,5)
```

## Fourier Transformation
The exact Algorithm is beyond the scope of this document, however the Youtube Channel [3b1b](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) provides an [excellent in-depth Explanation](https://www.youtube.com/watch?v=spUNpyF58BY).<br>
TLDR;<br>
A Fourier Transformation is a process by which the underlying frequencies of a noisy signal can be extracted.<br>

The [data](https://data.nationalgrideso.com/system/system-frequency-data?from=0#resources) has a resolution of one sample per second. So there is no need to calculate anything more precise than that since we would just be overfitting the data anyway.

```{code-cell} ipython3
from scipy.io import wavfile
from scipy.signal import resample

# change this to 60Hz for US-audio
enf = 50

# if you are using your own data, make sure its mono
fs_orig, data_orig = wavfile.read("ENF-WHU-Dataset/ENF-WHU-Dataset/H1/001.wav")
fs = 500
data = resample(data_orig, int((data_orig.shape[0] / fs_orig) * fs))
# the maximum frequency we want to analyze
# max_f = 50.5
# the precision with which we wish to analyze
# precision = 1/100
# the number of samples per second
# samplerate = np.ceil(2 * max_f)
# the number of samples per fft
# window_size = int(max_f / precision)

# data = resample(data_, int(data_.size * (samplerate / samplerate_)))#
```

## Applying a Bandpass
Since we are dealing with real-world data, the signal contains **a lot** more frequencies than just the ENF. Since we are not interested in any of them, we can remove them using a bandpass filter, that removes all frequencies outside a certain range. Luckily, scipy provides the [signal.iirpeak](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirpeak.html) filter, which is like a bandpass except for a very narrow band.

```{code-cell} ipython3
from scipy.signal import freqz, iirpeak


b, a = iirpeak(50, 30, fs=fs)

# test frequency response
freq, h = freqz(b, a, fs=fs)

# Convert frequency response to dB (logarithmic scale)
plt.plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)))
plt.title("Frequency Response")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.xlim([0, 100])
plt.ylim([-50, 10])
plt.axvspan(49, 51, facecolor="gray", alpha=0.5)
plt.xticks(list(plt.xticks()[0]) + [49, 51])
plt.grid()
plt.show()
```

Just slicing the wave file to the size of a window is insufficient, since it artificially creates edges at the start and end of the window. To avoid this, we multiply the window by a half cosine function to ensure a smooth transition. (This is called a [Hanning Window](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html))![hanning_window.png](attachment:hanning_window.png)

```{code-cell} ipython3
window = data[:window_size]
fourier = np.abs(np.fft.fft(window * np.hanning(window_size)))[:window_size//2]
frequencies = np.fft.fftfreq(2 * fourier.size, d=1/samplerate)[:window_size//2]

plt.plot(frequencies, fourier)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amount")
plt.title("Discrete time Fourier transform")
plt.show()
print("Main frequency at", np.argmax(fourier) * samplerate/window_size, "Hz")
```

## Improving DFT
The DFT above is flawed. There is a spike at around 0Hz that definitely does not belong there.
It has been shown that the accuracy of the Discrete-Time Fourier Transform increases when operating on the first-order derivative of the signal.

```{code-cell} ipython3
window_raw = data[:window_size]

window = window_raw * np.hanning(window_size)
window_derivative = np.diff(window_raw, prepend=0) * np.hanning(window_size)

fourier = np.abs(np.fft.fft(window))[:window_size//2]
fourier_derivative = np.abs(np.fft.fft(window_derivative))[:window_size//2]

frequencies = np.fft.fftfreq(2 * fourier.size, d=1/samplerate)[:window_size//2]

# scale |X'(K)| by F(K)
F = lambda x: (np.pi * x) / (window_size * np.sin((np.pi * x) / window_size))
fourier_derivative = F(np.arange(window_size//2)) * fourier_derivative

# remove the frequency of 0Hz - it does not make logical sense and messes up the F(0) calculation
fourier_derivative = fourier_derivative[1:]
frequencies = frequencies[1:]
fourier = fourier[1:]

# plt.plot(frequencies, fourier)
plt.plot(frequencies, fourier_derivative)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amount")
plt.title("Discrete time Fourier Transform (First-order derivative)")
plt.show()


f = 1/(2 * np.pi) * (np.argmax(fourier_derivative) / np.argmax(fourier))
print("Main frequency at", np.argmax(fourier_derivative) * samplerate/window_size, "Hz")
print(np.argmax(fourier_derivative))
print(f)
```

The weird spike at the beginning of the recording is gone and the spike at 48.76Hz is even clearer than before.

```{code-cell} ipython3
print(window_size, samplerate)
```

We can clearly see the spike at approximately 50Hz. This is the noise created by the electric network that we are looking
for. If, at this point, you do not see a spike, the audio does not contain that noise and it cannot be timestamped using the
algorithm.

+++

## Short-time Fourier Transform
This, however, is just a Fourier transform of the first second.<br>
Since we are interested in how the frequency varies over time, we need to compare multiple transformations against each other. This process is known as a [Short-time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)<br>
A samplerate of 48000 means that to get one Fourier transform per second, we need to offset our sliding window by 48000 samples each time. Since we expect a frequency of around 50 Hertz, a window size of 5 seconds, amounting to ~250 cycles should be more than sufficient.<br>
The next question is: How far should the sliding window shift after each iteration? This will determine the resolution of our final frequency graph. Since the [data](https://data.nationalgrideso.com/system/system-frequency-data?from=0#resources) only lists one data point per second, calculating anything more than that would just be overfitting the data. If you want to use this with your own dataset, you might want to adjust this value.

```{code-cell} ipython3
window_length = 10 # length of the window in seconds
window_size = samplerate * window_length # number of samples per window
n_windows = 1 # total number of windows
offset = 1024 # offset of each window
f_min = 49.5
f_max = 50.5
num_f = 100
d_f = (f_max - f_min) / num_f
result = np.empty((n_windows, num_f))
```

```{code-cell} ipython3
%%time
for ix in range(n_windows):
    window = data[ix * offset:ix * offset + window_size]
    window = window * np.hanning(window_size)
    fourier = np.abs(dft(window, d_f, np.linspace(f_min, f_max, num_f)))
    result[ix] = fourier[:num_f]
result = 20 * np.log10(result)
```

```{code-cell} ipython3
plt.imshow(result.transpose(), cmap="inferno", interpolation="nearest", aspect="auto", origin="lower")
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()
```

Now, is this just a slower, more complicated version of `plt.specgram`?<br>
Yes. But i feel like its worth understanding whats going on behind the scenes.

+++

We need to scale the values to the dB scale, which is logarithmic.

```{code-cell} ipython3
from scipy import signal
t, f, sxx = signal.spectrogram(data, fs=samplerate)

plt.pcolormesh(f, t, 10 * np.log10(sxx), shading="gouraud", cmap="inferno")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.show()
```

```{code-cell} ipython3
x = np.arange(100)
resampled = resample(x, 50)
print(resampled)
```

```{code-cell} ipython3
x = np.array([2, 3, 4, 1])
print(np.fft.fft(x))
print(np.fft.ifft(x))
```

```{code-cell} ipython3

```
