import numpy as np
import matplotlib.pyplot as plt


def g(theta):
    return ((1 + np.cos(theta) ** 2) / 2) ** 2 + np.cos(theta) ** 2


def to_xy(r, theta):
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y


def main():
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = g(theta)
    
    x, y = to_xy(r, theta)
    
    plt.rcParams.update({'text.usetex': True, 'font.size': 18})
    
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot(x, y, color='black')
    ax.set_aspect('equal')
    cmin, cmax = max(ax.get_xlim(),
                     ax.get_ylim(),
                     key=lambda c: c[1] - c[0])
    ax.set_xlim(cmin, cmax)
    ax.set_ylim(cmin, cmax)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    ax.set_yticks(list(range(-2, 3, 1)))
    ax.set_yticklabels([str(i) for i in range(-2, 3, 1)])
    
    fig.savefig('../images/angular_power_distribution.pdf')
    
    plt.show()
    return


if __name__ == "__main__":
    main()
