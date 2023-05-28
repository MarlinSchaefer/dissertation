import matplotlib.pyplot as plt
import numpy as np

MSUN = 2e30
G = 6.6e-11
C = 3e8
MPC = 3e22


def mchirp(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def phi(t, m1, m2, phi0=0):
    return -2 * (5 * G * mchirp(m1, m2) / C ** 3) ** (-5 / 8) * t ** (5 / 8) + phi0


def wave_prefactor(t, m1, m2, r):
    return (G * mchirp(m1, m2) / C ** 2) ** (5 / 4) / r * (5 / (C * t)) ** (1 / 4)


def hp(t, m1, m2, r, iota=0, theta=0, phi0=0):
    return wave_prefactor(t, m1, m2, r) * ((1 * np.cos(iota) ** 2) / 2) * np.cos(phi(t, m1, m2, phi0))


def hc(t, m1, m2, r, iota=0, theta=0, phi0=0):
    return wave_prefactor(t, m1, m2, r) * np.cos(iota) * np.cos(phi(t, m1, m2, phi0))


def main():
    m1 = 35 * MSUN
    m2 = 30 * MSUN
    r = 1000 * MPC
    t = np.linspace(0.01, 5, 100000)
    y = hp(t, m1, m2, r)
    
    plt.rcParams.update({'text.usetex': True})
    
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 3.6))
    ax.plot(-t, y, c='black')
    ax.set_xlabel('Time until merger [s]')
    ax.set_ylabel('Strain')
    
    ax.set_xticks([-i for i in range(6)])
    ax.set_xticklabels(list(range(6)))
    
    fig.savefig('../images/linear_waveform.pdf', bbox_inches='tight')
    plt.show()
    return


if __name__ == "__main__":
    main()
