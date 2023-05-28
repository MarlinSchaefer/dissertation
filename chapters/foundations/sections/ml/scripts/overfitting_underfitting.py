import numpy as np
import matplotlib.pyplot as plt

SEED = 3953629938
ACTUAL_DEGREE = 2
UNDER_DEGREE = 1
OVER_DEGREE = 9
RANGE = [-4, 4]


def mse(y1, y2):
    return np.mean(np.square(y1 - y2))


def plot_axes(ax, x, y, xt, yt, xv, yv, title=None, markersize=150):
    ax.plot(x, y, zorder=3)
    ax.scatter(xt, yt, edgecolor='black', zorder=5, label='Train set',
               s=markersize)
    ax.scatter(xv, yv, edgecolor='black', zorder=4, marker='s',
               label='Validation set', s=markersize)
    if title is not None:
        ax.set_title(title)


def set_ticklabelsize(ax, labelsize):
    for pt in ax.get_xticklabels():
        pt.set_fontsize(labelsize)
    for pt in ax.get_yticklabels():
        pt.set_fontsize(labelsize)


def main():
    if SEED is None:
        rs = np.random.RandomState(SEED)
        seed = rs.randint(0, 2**32 - 1)
    else:
        seed = SEED
    rs = np.random.RandomState(seed)
    coef = rs.randint(-10, 10, ACTUAL_DEGREE + 1)
    print(f"Seed: {seed}")
    print(coef)
    if coef[-1] == 0:
        coef[-1] = 1
    poly = np.polynomial.Polynomial(coef)
    xs = rs.uniform(*RANGE, OVER_DEGREE-4)
    ys = poly(xs)
    
    xv = np.array([-3.8, 0, 2.5])
    yv = poly(xv)
    
    overfit = np.polynomial.Polynomial.fit(xs, ys, OVER_DEGREE)
    underfit = poly.fit(xs, ys, UNDER_DEGREE)
    
    print(overfit.coef)
    
    # x = np.linspace(*RANGE, 1000)
    x = np.linspace(*RANGE, 1000)
    y = poly(x)
    overy = overfit(x)
    undery = underfit(x)
    
    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(19.2, 10.8))
    fontsize = 28
    plt.rc('font', size=fontsize)
    plt.rc('text', usetex=True)
    ticklabelsize = 20
    # plt.rc('xtick', labelsize=300)
    
    for ax in axs:
        ax.grid()
    
    np.set_printoptions(suppress=True)
    # Underfit
    title = (f'Train MSE: {mse(ys, underfit(xs)):.2f}\n'
             f'Validation MSE: {mse(yv, underfit(xv)):.2f}')
    plot_axes(axs[0], x, undery, xs, ys, xv, yv, title=title)
    set_ticklabelsize(axs[0], ticklabelsize)
    
    # Actual
    title = (f'Train MSE: {mse(ys, poly(xs)):.2f}\n'
             f'Validation MSE: {mse(yv, poly(xv)):.2f}')
    plot_axes(axs[1], x, y, xs, ys, xv, yv, title=title)
    set_ticklabelsize(axs[1], ticklabelsize)
    
    axpoint = (0.05, 0.99)
    ax = axs[1]
    dT = ax.transData.inverted()
    dA = ax.transAxes
    point = dT.transform(dA.transform(axpoint))
    
    # Overfit
    title = (f'Train MSE: {mse(ys, overfit(xs)):.2f}\n'
             f'Validation MSE: {mse(yv, overfit(xv)):.2f}')
    plot_axes(axs[2], x, overy, xs, ys, xv, yv, title=title)
    set_ticklabelsize(axs[2], ticklabelsize)
    
    axs[0].text(*point, '(a)', fontsize=fontsize, ha='left', va='top')
    axs[1].text(*point, '(b)', fontsize=fontsize, ha='left', va='top')
    axs[2].text(*point, '(c)', fontsize=fontsize, ha='left', va='top')
    
    axs[2].legend(fontsize=fontsize)
    
    fig.tight_layout()
    
    fig.savefig('../images/overunderfit.pdf')
        
    plt.show()


if __name__ == "__main__":
    main()
