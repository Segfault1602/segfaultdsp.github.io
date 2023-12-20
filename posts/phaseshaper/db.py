import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

FS = 48000
F0 = 1000

t = np.linspace(0, 1, FS)
s = np.sin(2 * np.pi * F0 * t)

NFFT = len(s)
sf = fft.fft(s, NFFT)
sf = fft.fftshift(sf)
sf = np.log10(np.abs(sf)) - 5
freq = fft.fftfreq(NFFT, 1 / FS)
freq = fft.fftshift(freq)

fig, axs = plt.subplots(2, 1)
top = axs[0]
top.plot(t, s)

bot = axs[1]
bot.magnitude_spectrum(s, Fs=FS)
# plt.specgram(s, Fs=FS, NFFT=NFFT, mode="psd", scale="dB")
# bot.plot(freq, sf)
# bot.set_xlim(0, 2000)
plt.show()
