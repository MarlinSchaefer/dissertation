import numpy as np
import matplotlib.pyplot as plt


def equidistant_angle(n):
    return np.arange(n) * 2 * np.pi / n


def circle_points(n, r=1):
    phi = equidistant_angle(n)
    return r * np.cos(phi), r * np.sin(phi)


def hp(t, A=1):
    return A * np.sin(t)


def hc(t, A=1):
    return A * np.sin(t)


def change(x, y, t, Ap=1, Ac=1):
    hpv = hp(t, A=Ap)
    hcv = hc(t, A=Ac)
    return (1 + hpv / 2) * x + hcv * y / 2, (1 - hpv / 2) * y + hcv / 2 * x


def main():
    t = equidistant_angle(4)
    x, y = circle_points(8)
    xc, yc = circle_points(1000)
    
    basesize = 4
    
    w_weights = [0.2] + [1] * len(t)
    h_weights = [0.2] + [1] * 2
    plt.rcParams.update({'text.usetex': True,
                         'font.size': 26,
                         'axes.linewidth': 4})
    fig = plt.figure(figsize=(sum(w_weights) * basesize,
                              sum(h_weights) * basesize))
    gridspec = fig.add_gridspec(ncols=len(w_weights),
                                nrows=len(h_weights),
                                width_ratios=w_weights,
                                height_ratios=h_weights,
                                hspace=0,
                                wspace=0)
    
    top_labels = ['$0$', '$\\frac{\\pi}{2}$', '$\\pi$', '$\\frac{3\\pi}{2}$']
    for i in range(1, len(w_weights)):
        ax = fig.add_subplot(gridspec[0, i])
        ax.text(0, 0, top_labels[i-1], ha='center', va='center')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    left_labels = ['$h_+$', '$h_\\times$']
    for i in range(1, len(h_weights)):
        ax = fig.add_subplot(gridspec[i, 0])
        ax.text(0, 0, left_labels[i-1], ha='center', va='center')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    rows, cols = len(h_weights), len(w_weights)
    for row in range(rows - 1):
        row += 1
        ap, ac = (1, 0) if row % 2 == 1 else (0, 1)
        for col in range(cols - 1):
            col += 1
            ax = fig.add_subplot(gridspec[row, col])
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.plot(*change(xc, yc, t[col-1], Ap=ap, Ac=ac),
                    ls='--',
                    c='black')
            ax.scatter(*change(x, y, t[col-1], Ap=ap, Ac=ac),
                       c='black')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    
    fig.savefig('../images/effect_gw.pdf', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
