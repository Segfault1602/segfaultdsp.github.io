"""_summary_

https://mural.maynoothuniversity.ie/4096/1/vps_dafx11.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def lissajou(a, b, theta, wd, wv):
    d = a * (0.5 + 0.5 * np.cos(wd + theta))
    v = b * (0.5 + 0.5 * np.cos(wv))

    return d, v


FS = 48000

fw = 3
fd = 1

a = 1
b = 3
theta = np.pi / 4
t = np.linspace(0, 1, FS)

wd = 2 * np.pi * fd * t
wv = 2 * np.pi * fw * t


d, v = lissajou(a, b, theta, wd, wv)

plt.plot(d, v)
plt.show()
