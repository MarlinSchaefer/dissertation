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


def F(theta, phi):
    fp = (1 + np.cos(theta)**2) / 2 * np.sin(2*phi)
    fc = np.cos(theta) * np.cos(2 * phi)
    return fp, fc


def project(hpw, hcw, theta, phi, psi):
    hpd = np.cos(2 * psi) * hpw - np.sin(2 * psi) * hcw
    hcd = np.sin(2 * psi) * hpw + np.cos(2 * psi) * hcw
    fp, fc = F(theta, phi)
    return fp * hpd + fc * hcd


def main():
    m1 = 35 * MSUN
    m2 = 30 * MSUN
    r = 1000 * MPC
    t = np.linspace(0.01, 5, 1000)
    L = 4e3
    dec = 0
    ra = 0
    pol = 0
    
    hpw = hp(t, m1, m2, r)
    hcw = hc(t, m1, m2, r)
    h = project(hpw, hcw, dec, ra, pol)
    print(max(h / 2 * L))


if __name__ == "__main__":
    main()
