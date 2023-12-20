"""_summary_

https://mural.maynoothuniversity.ie/4096/1/vps_dafx11.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.mlab as mlab
from scipy.signal.windows import chebwin
from scipy.fft import fft


FS = 48000


def spectrum(s):
    NFFT = 16384 * 2
    NWIN = 16384
    win = chebwin(NWIN, 120)
    win = np.append(win, np.zeros(NFFT - NWIN))
    scal = NFFT * np.sqrt(np.mean(win**2))
    spec = fft(win * s[0:NFFT])
    mags = np.sqrt(
        spec[0 : int(NFFT / 2)].real ** 2 + spec[0 : int(NFFT / 2)].imag ** 2
    )
    norm = 20 * np.log10(mags / scal)
    spec = norm - max(norm)
    return spec


### Classic Phase Distortion (Section 2)
def phase_distort(x, d):
    return 0.5 * (x / d) if x < d else 0.5 * (1 + (x - d) / (1 - d))


class ClassicPhaseDistortion:
    def __init__(self, f0, axes):
        self.f0 = f0
        self.tdata = np.linspace(0, 1, FS)
        self.axes = axes

        s1 = self.tdata * f0
        self.phi = np.array([np.mod(x, 1) for x in s1])
        (self.line_phi,) = self.axes[0].plot(self.tdata, self.phi, label="Phase")

        self.signal = -np.cos(2 * np.pi * self.phi)
        (self.line_signal,) = self.axes[0].plot(
            self.tdata, self.signal, label="Waveform"
        )
        self.axes[0].set_xlim(0, 2 / f0)
        self.axes[0].grid(True)

        spec, freqs = mlab.magnitude_spectrum(self.signal, Fs=FS)
        (self.spectrum,) = self.axes[1].plot(freqs, spec)
        self.axes[1].set_xlim(0, FS / 2)

    def update(self, i):
        phi_distorted = np.array([phase_distort(x, i) for x in self.phi])
        s = -np.cos(2 * np.pi * phi_distorted)

        self.line_phi.set_data(self.tdata, phi_distorted)
        self.line_signal.set_data(self.tdata, s)

        spec, freqs = mlab.magnitude_spectrum(s, Fs=FS)
        self.spectrum.set_data(freqs, spec)
        return (
            self.line_phi,
            self.line_signal,
            self.spectrum,
        )


def classic_pd():
    F0 = 5000
    fig, axs = plt.subplots(2, 1)
    pd = ClassicPhaseDistortion(F0, axs)
    frames_ani = np.concatenate(
        [np.linspace(0.01, 0.99, 300), np.linspace(0.99, 0.01, 300)]
    )
    ani = animation.FuncAnimation(
        fig, pd.update, frames=frames_ani, interval=1, blit=True
    )

    plt.show()


# Section 3
# Vector Phaseshaping (Figure 2)


def phase_distort_vec(x, d, v):
    return (v * x / d) if x < d else (1 - v) * ((x - d) / (1 - d)) + v


class VectorPhaseshaping:
    def __init__(self, f0, axes):
        self.f0 = f0
        self.tdata = np.linspace(0, 1, FS)
        self.axes = axes

        s1 = self.tdata * f0
        self.phi = np.array([np.mod(x, 1) for x in s1])
        (self.line_phi,) = self.axes[0].plot(self.tdata, self.phi, label="Phase")

        self.signal = -np.cos(2 * np.pi * self.phi)
        (self.line_signal,) = self.axes[0].plot(
            self.tdata, self.signal, label="Waveform"
        )
        self.axes[0].set_xlim(0, 2 / f0)
        self.axes[0].grid(True)

        spec, freqs = mlab.magnitude_spectrum(self.signal, Fs=FS)
        (self.spectrum,) = self.axes[1].plot(freqs, spec)
        self.axes[1].set_xlim(0, FS / 2)

    def update(self, i):
        phi_distorted = np.array([phase_distort_vec(x, 0.5, i) for x in self.phi])
        s = -np.cos(2 * np.pi * phi_distorted)

        self.line_phi.set_data(self.tdata, phi_distorted)
        self.line_signal.set_data(self.tdata, s)

        spec, freqs = mlab.magnitude_spectrum(s, Fs=FS)
        self.spectrum.set_data(freqs, spec)
        return (
            self.line_phi,
            self.line_signal,
            self.spectrum,
        )


def vector_phaseshaping():
    F0 = 500
    fig, axs = plt.subplots(2, 1)
    pd = VectorPhaseshaping(F0, axs)
    frames_ani = np.concatenate(
        [np.linspace(0.01, 0.99, 300), np.linspace(0.99, 0.01, 300)]
    )
    ani = animation.FuncAnimation(
        fig, pd.update, frames=frames_ani, interval=1, blit=True
    )

    plt.show()


# # figure 3


class VectorPhaseshapingPulseWidth:
    def __init__(self, f0, axes):
        self.f0 = f0
        self.tdata = np.linspace(0, 1, FS)
        self.axes = axes

        s1 = self.tdata * f0
        self.phi = np.array([np.mod(x, 1) for x in s1])
        (self.line_phi,) = self.axes[0].plot(self.tdata, self.phi, label="Phase")

        self.signal = -np.cos(2 * np.pi * self.phi)
        (self.line_signal,) = self.axes[0].plot(
            self.tdata, self.signal, label="Waveform"
        )
        self.axes[0].set_xlim(0, 2 / f0)
        self.axes[0].grid(True)

        spec, freqs = mlab.magnitude_spectrum(self.signal, Fs=FS)
        (self.spectrum,) = self.axes[1].plot(freqs, spec)
        self.axes[1].set_xlim(0, FS / 2)

    def update(self, i):
        phi_distorted = np.array([phase_distort_vec(x, i, 1) for x in self.phi])
        s = -np.cos(2 * np.pi * phi_distorted)

        self.line_phi.set_data(self.tdata, phi_distorted)
        self.line_signal.set_data(self.tdata, s)

        spec, freqs = mlab.magnitude_spectrum(s, Fs=FS)
        self.spectrum.set_data(freqs, spec)
        return (
            self.line_phi,
            self.line_signal,
            self.spectrum,
        )


def vector_phaseshaping_pulse():
    F0 = 500
    fig, axs = plt.subplots(2, 1)
    pd = VectorPhaseshapingPulseWidth(F0, axs)
    frames_ani = np.concatenate(
        [np.linspace(0.05, 0.50, 80), np.linspace(0.50, 0.05, 80)]
    )
    ani = animation.FuncAnimation(
        fig, pd.update, frames=frames_ani, interval=1, blit=True
    )

    plt.show()


class VpsFormant:
    def __init__(self, f0, d, axes):
        self.d = d
        self.f0 = f0
        self.tdata = np.linspace(0, 1, FS)
        self.axes = axes

        s1 = self.tdata * f0
        self.phi = np.array([np.mod(x, 1) for x in s1])
        (self.line_phi,) = self.axes[0].plot(self.tdata, self.phi, label="Phase")

        self.signal = -np.cos(2 * np.pi * self.phi)
        (self.line_signal,) = self.axes[0].plot(
            self.tdata, self.signal, label="Waveform"
        )
        self.axes[0].set_xlim(0, 1.2 / f0)
        self.axes[0].grid(True)

        spec = spectrum(self.signal)
        (self.spectrum,) = self.axes[1].plot(spec, lw=1)
        self.axes[1].set_xlim(0, FS / 2)
        self.axes[1].set_ylim(-80, 6)

    def update(self, i):
        phi_distorted = np.array([phase_distort_vec(x, self.d, i) for x in self.phi])
        s = -np.cos(2 * np.pi * phi_distorted)

        self.line_phi.set_data(self.tdata, phi_distorted)
        self.line_signal.set_data(self.tdata, s)

        spec = spectrum(s)

        # spec, freqs = mlab.magnitude_spectrum(y, Fs=FS)
        self.spectrum.set_ydata(spec)
        return (
            self.line_phi,
            self.line_signal,
            self.spectrum,
        )


def vps_formant():
    F0 = 2500
    fig, axs = plt.subplots(2, 2)
    ax1 = [axs[0, 0], axs[1, 0]]
    pd = VpsFormant(F0, 0.5, ax1)
    f = np.linspace(1, 5.5, 100)
    frames_ani = np.concatenate([f, f[-2::-1]])
    ani = animation.FuncAnimation(
        fig, pd.update, frames=frames_ani, interval=50, blit=True
    )
    axs[0, 0].set_ylim(-1, 5)
    axs[0, 1].set_ylim(-1, 5)

    # fig2, axs2 = plt.subplots(2, 1)
    ax2 = [axs[0, 1], axs[1, 1]]
    pd2 = VpsFormant(F0, 0.25, ax2)
    f2 = np.linspace(1, 5.5, 100)
    frames_ani2 = np.concatenate([f2, f2[-2::-1]])
    ani2 = animation.FuncAnimation(
        fig, pd2.update, frames=frames_ani2, interval=50, blit=True
    )
    # axs2[0].set_ylim(-1, 5)
    plt.show()


def phase_distort_aa(x, d, v):
    p = phase_distort_vec(x, d, v)
    b = np.mod(v, 1)

    if p > int(v):
        if b <= 0.5:
            p = 0.5 * np.mod(p, 1) / b
        else:
            p = np.mod(p, 1) / b

    return p


def scale_vps(st, p, c, v):
    if p > int(v):
        if c < 0.5:
            c1 = 0.5 * (1 - np.cos(2 * np.pi * c))
            return c1 * st + (c1 - 1)
        else:
            c1 = 0.5 * (1 + np.cos(2 * np.pi * c))
            return c1 * st + (1 - c1)
    return st


class VpsFormantAntiAlias:
    def __init__(self, f0, d, axes):
        self.d = d
        self.f0 = f0
        self.tdata = np.linspace(0, 1, FS)
        self.axes = axes

        s1 = self.tdata * f0
        self.phi = np.array([np.mod(x, 1) for x in s1])
        (self.line_phi,) = self.axes[0].plot(self.tdata, self.phi, label="Phase")

        self.signal = -np.cos(2 * np.pi * self.phi)
        (self.line_signal,) = self.axes[0].plot(
            self.tdata, self.signal, label="Waveform"
        )
        self.axes[0].set_xlim(0, 1.2 / f0)
        self.axes[0].grid(True)

        # spec, freqs = mlab.magnitude_spectrum(self.signal, Fs=FS)
        spec = spectrum(self.signal)
        (self.spectrum,) = self.axes[1].plot(spec, lw=1)
        self.axes[1].set_xlim(0, FS / 2)
        self.axes[1].set_ylim(-80, 6)

    def update(self, i):
        phi_2 = np.array([phase_distort_vec(x, self.d, i) for x in self.phi])
        phi_distorted = np.array([phase_distort_aa(x, self.d, i) for x in self.phi])
        y = -np.cos(2 * np.pi * phi_distorted)
        b = np.mod(i, 1)

        y = np.array([scale_vps(x, p, b, i) for (x, p) in zip(y, phi_2)])

        self.line_phi.set_data(self.tdata, phi_distorted)
        self.line_signal.set_data(self.tdata, y)

        spec = spectrum(y)

        # spec, freqs = mlab.magnitude_spectrum(y, Fs=FS)
        self.spectrum.set_ydata(spec)
        return (
            self.line_phi,
            self.line_signal,
            self.spectrum,
        )


def vps_formant_aa():
    F0 = 500
    fig, axs = plt.subplots(2, 2)
    ax1 = [axs[0, 0], axs[1, 0]]
    pd = VpsFormant(F0, 0.5, ax1)
    f = np.linspace(1, 5.5, 100)
    frames_ani = np.concatenate([f, f[-2::-1]])
    ani = animation.FuncAnimation(
        fig, pd.update, frames=frames_ani, interval=33, blit=True
    )
    axs[0, 0].set_ylim(-1, 5)
    axs[0, 1].set_ylim(-1, 5)

    # fig2, axs2 = plt.subplots(2, 1)
    ax2 = [axs[0, 1], axs[1, 1]]
    pd2 = VpsFormantAntiAlias(F0, 0.25, ax2)
    f2 = np.linspace(1, 5.5, 500)
    frames_ani2 = np.concatenate([f2, f2[-2::-1]])
    ani2 = animation.FuncAnimation(
        fig, pd2.update, frames=frames_ani, interval=33, blit=True
    )
    # axs2[0].set_ylim(-1, 5)
    plt.show()


if __name__ == "__main__":
    # classic_pd()
    # vector_phaseshaping()
    # vector_phaseshaping_pulse()
    # vps_formant()
    vps_formant_aa()
