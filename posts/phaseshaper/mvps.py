"""_summary_

https://mural.maynoothuniversity.ie/4096/1/vps_dafx11.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

FS = 48000


def phase_distort_vec(x, p):
    try:
        p1 = next(i for i, (d, v) in enumerate(p) if x < d)
    except StopIteration:
        d, v = p[-1]
        return (1 - v) * (x - d) / (1 - d) + v

    dn, vn = p[p1]
    dn1, vn1 = p[p1 - 1]

    if p1 == 0:
        return vn * x / dn

    return (vn - vn1) * (x - dn1) / (dn - dn1) + vn1


d = [0.1, 0.5, 0.6]
v = [0.5, 0.5, 1]

p = list(zip(d, v))

phi = np.linspace(0, 1, 1000)

phi = np.array([phase_distort_vec(x, p) for x in phi])
y = -np.cos(2 * np.pi * phi)

plt.plot(phi)
plt.plot(y)
plt.show()
