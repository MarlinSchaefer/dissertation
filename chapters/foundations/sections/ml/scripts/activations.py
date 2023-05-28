import matplotlib.pyplot as plt
import numpy as np


def ReLU(x):
    return np.maximum(x, 0)


def dReLU(x):
    return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))


def Tanh(x):
    return np.tanh(x)


def dTanh(x):
    return 1 - Tanh(x) ** 2


def ELU(x, alpha=1):
    return np.where(x >= 0, x, alpha * np.exp(x) - 1)


def dELU(x, alpha=1):
    return np.where(x >= 0, 1, alpha * np.exp(x))


def Linear(x):
    return x


def dLinear(x):
    return np.ones_like(x)


def Heaviside(x):
    return (x > 0).astype(np.int)


def dHeaviside(x):
    return np.zeros_like(x)


def set_ticklabelsize(ax, labelsize):
    for pt in ax.get_xticklabels():
        pt.set_fontsize(labelsize)
    for pt in ax.get_yticklabels():
        pt.set_fontsize(labelsize)


def set_tickfont(ax, font):
    ax.set_xticklabels(ax.get_xticks(), font)
    ax.set_yticklabels(ax.get_yticks(), font)


def main():
    fontsize = 36
    lw = 4
    ticklabelsize = 28
    x = np.linspace(-5, 5, 1000)
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(19.2 * 2, 10.8))
    plt.rcParams.update({'font.size': fontsize,
                         'text.usetex': True,
                         'font.family': 'serif'})
    fontdict = {'size': fontsize, 'family': 'serif'}
    tickfont = {'size': ticklabelsize, 'family': 'serif'}
    # plt.rc('font', size=fontsize)
    # plt.rc('text', usetex=True)
    
    # Plot activations
    ax = axs[0]
    for func in [Heaviside, Sigmoid, Tanh, ELU, ReLU]:
        y = func(x)
        ax.plot(x, y, label=func.__name__, lw=lw)
    ax.plot(x, Linear(x), label='Linear', ls='--', color='black', lw=lw)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Input', font=fontdict)
    ax.set_ylabel('Output', font=fontdict)
    ax.legend()
    ax.grid()
    ax.set_title('Activation functions', font=fontdict)
    set_ticklabelsize(ax, ticklabelsize)
    # set_tickfont(ax, tickfont)
    
    # Plot derivatives
    ax = axs[1]
    for func in [dHeaviside, dSigmoid, dTanh, dELU, dReLU]:
        y = func(x)
        ax.plot(x, y, lw=lw)
    ax.plot(x, dLinear(x), ls='--', color='black', lw=lw)
    ax.set_xlabel('Input', font=fontdict)
    ax.grid()
    ax.set_title('Derivatives')
    set_ticklabelsize(ax, ticklabelsize)
    # set_tickfont(ax, tickfont)
    
    # Save figure
    fig.tight_layout()
    fig.savefig('../images/activations.pdf')
    plt.show()
    return


if __name__ == '__main__':
    main()
