import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import ligo.skymap.plot


def Fp(theta, phi):
    return (1 + np.cos(theta)**2) / 2 * np.sin(2*phi)


def Fc(theta, phi):
    return np.cos(theta) * np.sin(2*phi)
    

def npix(nside):
    return 12 * nside ** 2


def main():
    nside = 16
    theta, phi = hp.pix2ang(nside, np.arange(npix(nside)))
    
    res = Fp(theta, phi)**2 + Fc(theta, phi)**2
    
    fig, ax = plt.subplots(ncols=1,
                           figsize=(2 * 4.8, 3.6),
                           subplot_kw={'projection':
                                       'astro degrees mollweide'})
    im = ax.imshow_hpx(res, cmap='cylon', vmin=0, vmax=2)
    plt.colorbar(im)
    ax.grid()
    
    fig.savefig('../images/sky_location_power.pdf')
    plt.close('all')
    return


if __name__ == '__main__':
    main()
