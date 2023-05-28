import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as Line
from PIL import Image
from scipy import signal


def get_kernel_vertical(channels=3):
    kern = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])
    if channels is None:
        return kern
    return np.stack([kern for _ in range(channels)]).transpose(1, 2, 0)


def get_kernel_horizontal(channels=3):
    if channels is None:
        return get_kernel_vertical(channels=channels).T
    return get_kernel_vertical(channels=channels).transpose(1, 0, 2)


def convolve(img, kernel):
    if img.ndim == 2:
        return signal.convolve2d(img, kernel, mode='same')
    ret = np.zeros_like(img)
    for channel in range(img.shape[2]):
        ret[:, :, channel] = signal.convolve2d(img[:, :, channel],
                                               kernel[:, :, channel],
                                               mode="same")
    return ret


def draw_kernel(ax, kernel, style=None):
    if style is None:
        style = {}
    grid_style = {'c': 'black'}
    grid_style.update(style)
    ax.plot([0, kernel.shape[1]], [0, 0], **grid_style)
    ax.plot([0, 0], [0, kernel.shape[0]], **grid_style)
    for rowidx, row in enumerate(reversed(kernel)):
        ax.plot([0, kernel.shape[1]], [rowidx + 1, rowidx + 1], **grid_style)
        for colidx, value in enumerate(row):
            ax.plot([colidx + 1, colidx + 1], [0, kernel.shape[0]],
                    **grid_style)
            ax.text(colidx + 0.5, rowidx + 0.5, str(value),
                    va='center', ha='center')
    ax.set_aspect('equal')


def get_transforms(ax):
    fromD = ax.transData
    toD = fromD.inverted()
    fromAx = ax.transAxes
    toAx = fromAx.inverted()
    return toD.transform, fromD.transform, toAx.transform, fromAx.transform


def center_above_line(c1, c2, sep=0.1):
    d1 = c2[0] - c1[0]
    d2 = c2[1] - c1[1]
    mid = np.array([d1 / 2, d2 / 2])
    ort = np.array([-mid[1], mid[0]]) / np.sqrt(np.sum(np.square(mid))) * sep
    return c1[0] + mid[0] + ort[0], c1[1] + mid[1] + ort[1]


def main():
    # Load image
    img = Image.open('../images/landscape.jpg')
    imgarr = np.asarray(img)
    
    # Initialize kernels
    vkernel = get_kernel_vertical(channels=None)
    hkernel = get_kernel_horizontal(channels=None)
    
    # Apply kernels
    vimg = convolve(imgarr, vkernel)
    himg = convolve(imgarr, hkernel)
    
    # Plotting
    fig = plt.figure(figsize=(2.4 * 4.8 * 1.5, 2 * 3.6 * 1.5))
    gridspec = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[1, 0.4, 1],
                                wspace=0.5)
    axs = []
    
    ax = fig.add_subplot(gridspec[:, 0])
    ax.imshow(imgarr, cmap='gray', vmin=0, vmax=255)
    ax.set_axis_off()
    axs.append(ax)
    
    ax = fig.add_subplot(gridspec[0, 1])
    draw_kernel(ax, vkernel)
    ax.set_axis_off()
    axs.append(ax)
    
    ax = fig.add_subplot(gridspec[1, 1])
    draw_kernel(ax, hkernel)
    ax.set_axis_off()
    axs.append(ax)
    
    ax = fig.add_subplot(gridspec[0, 2])
    ax.imshow(vimg, cmap='gray', vmin=0, vmax=255)
    ax.set_axis_off()
    axs.append(ax)
    
    ax = fig.add_subplot(gridspec[1, 2])
    ax.imshow(himg, cmap='gray', vmin=0, vmax=255)
    ax.set_axis_off()
    axs.append(ax)
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set(figheight=bbox.height, figwidth=bbox.width)
    
    # fig.savefig('../images/edge_filter.png',
    #             bbox_inches='tight',
    #             pad_inches=0)
    
    # Create lines and text between different subplots
    # Setup transforms
    fromFig = fig.transFigure
    toFig = fromFig.inverted().transform
    fromFig = fromFig.transform
    
    toD1, fromD1, toAx1, fromAx1 = get_transforms(axs[0])
    toD2, fromD2, toAx2, fromAx2 = get_transforms(axs[1])
    toD3, fromD3, toAx3, fromAx3 = get_transforms(axs[2])
    toD4, fromD4, toAx4, fromAx4 = get_transforms(axs[3])
    toD5, fromD5, toAx5, fromAx5 = get_transforms(axs[4])
    
    # Draw lines
    c1 = toFig(fromAx1([1, 0.8]))
    c2 = toFig(fromD2([0, 1.5]))
    line = Line((c1[0], c2[0]), (c1[1], c2[1]),
                transform=fig.transFigure,
                ls='dashed',
                c='black')
    fig.lines.append(line)
    tc = center_above_line(c1, c2, sep=0.02)
    fig.text(*tc, '$\\ast$', va='center', ha='center')
    
    c1 = toFig(fromAx1([1, 0.2]))
    c2 = toFig(fromD3([0, 1.5]))
    line = Line((c1[0], c2[0]), (c1[1], c2[1]),
                transform=fig.transFigure,
                ls='dashed',
                c='black')
    fig.lines.append(line)
    tc = center_above_line(c1, c2, sep=-0.02)
    fig.text(*tc, '$\\ast$', va='center', ha='center')
    
    c1 = toFig(fromD2([3, 1.5]))
    c2 = toFig(fromAx4([0, 0.5]))
    line = Line((c1[0], c2[0]), (c1[1], c2[1]),
                transform=fig.transFigure,
                ls='dashed',
                c='black')
    fig.lines.append(line)
    
    c1 = toFig(fromD3([3, 1.5]))
    c2 = toFig(fromAx5([0, 0.5]))
    line = Line((c1[0], c2[0]), (c1[1], c2[1]),
                transform=fig.transFigure,
                ls='dashed',
                c='black')
    fig.lines.append(line)
    
    # Save figure
    fig.savefig('../images/edge_filter.pdf',
                bbox_inches='tight',
                pad_inches=0)
    plt.show()
    return


if __name__ == "__main__":
    main()
