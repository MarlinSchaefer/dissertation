import matplotlib.pyplot as plt
import numpy as np

from pycbc.catalog import Merger
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.psd import interpolate

WINDOW = (1., -0.45)
FSAMP = 512


def restrict_frequency(freqser, upper=None, lower=None):
    if lower is None:
        lower = min(freqser.sample_frequencies)
    if upper is None:
        upper = max(freqser.sample_frequencies)
    lidx = np.searchsorted(freqser.sample_frequencies,
                           lower,
                           side='left')
    ridx = np.searchsorted(freqser.sample_frequencies,
                           upper,
                           side='right')
    return freqser.sample_frequencies[lidx:ridx], freqser.data[lidx:ridx]


def main():
    merger = Merger('GW150914')
    
    strain = merger.strain('H1')
    mtime = (float(strain.end_time) - float(strain.start_time)) / 2
    strain.start_time = 0
    psd1 = strain.psd(1)
    mid = float(strain.end_time) / 2
    short_strain = strain.time_slice(mid - WINDOW[0], mid + WINDOW[1])
    short_strain.start_time = 0
    
    plt.rcParams.update({'text.usetex': True,
                         'font.size': 16})
    
    # Plot raw
    fig, axs = plt.subplots(ncols=2, figsize=(2 * 4.8, 3.6))
    plt.subplots_adjust(wspace=0.5)
    ax = axs[0]
    ax.plot(short_strain.sample_times, short_strain)
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Strain [1]')
    ax.set_title('Detector strain')
    
    ax = axs[1]
    # ax.semilogy(*restrict_frequency(psd1**0.5, upper=1024))
    ax.loglog(*restrict_frequency(psd1**0.5, upper=1024))
    ax.set_xlim(10, ax.get_xlim()[1])
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude spectral density [$1/\\sqrt{Hz}$]')
    ax.set_title('Amplitude spectral density')
    
    fig.savefig('../images/data_raw.pdf', bbox_inches='tight')
    
    plt.clf()
    
    # Plot whitened
    white = strain.whiten(1, 1)
    white.start_time = 0
    psd2 = white.psd(1)
    mid = float(white.end_time) / 2
    white_short = white.time_slice(mid - WINDOW[0], mid + WINDOW[1])
    white_short.start_time = 0
    
    fig, axs = plt.subplots(ncols=2, figsize=(2 * 4.8, 3.6))
    plt.subplots_adjust(wspace=0.5)
    ax = axs[0]
    ax.plot(white_short.sample_times, white_short)
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Strain [1]')
    ax.set_title('Whitened strain')
    
    ax = axs[1]
    # ax.semilogy(*restrict_frequency(psd2**0.5, upper=1024))
    ax.loglog(*restrict_frequency(psd2**0.5, upper=1024))
    ax.set_xlim(10, ax.get_xlim()[1])
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude spectral density [$1/\\sqrt{Hz}$]')
    ax.set_title('Amplitude spectral density')
    
    fig.savefig('../images/data_white.pdf', bbox_inches='tight')
    
    plt.clf()
    
    # Plot bandpassed
    bandpassed = white.lowpass_fir(256, FSAMP).highpass_fir(20, FSAMP)
    psd3 = bandpassed.psd(1)
    mid = float(bandpassed.end_time) / 2
    dsamp = FSAMP * bandpassed.delta_t
    bandpassed_short = bandpassed.time_slice(mid - WINDOW[0] + dsamp,
                                             mid + WINDOW[1] + dsamp)
    bandpassed_short.start_time = 0
    
    fig, axs = plt.subplots(ncols=2, figsize=(2 * 4.8, 3.6))
    plt.subplots_adjust(wspace=0.5)
    ax = axs[0]
    ax.plot(bandpassed_short.sample_times, bandpassed_short)
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Strain [1]')
    ax.set_title('Bandpassed whitened strain')
    
    ax = axs[1]
    # ax.semilogy(*restrict_frequency(psd3**0.5, upper=1024))
    ax.loglog(*restrict_frequency(psd3**0.5, upper=1024))
    ax.set_xlim(10, ax.get_xlim()[1])
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude spectral density [$1/\\sqrt{Hz}$]')
    ax.set_title('Amplitude spectral density')
    
    fig.savefig('../images/data_bandpassed.pdf', bbox_inches='tight')
    
    plt.clf()
    return


if __name__ == "__main__":
    main()
