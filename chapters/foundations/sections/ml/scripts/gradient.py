import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from mayavi import mlab
from matplotlib.cm import get_cmap # for viridis


def Arrow(src, dst):
    chg = dst - src
    x, y, z = [np.array([[[pt]]]) for pt in src]
    u, v, w = [np.array([[[pt]]]) for pt in chg]
    return mlab.quiver3d(x, y, z, u, v, w,
                         line_width=1, color=(1, 0, 0), colormap='hsv', 
                         scale_factor=1, mode='arrow',resolution=10)


def normalize(vec):
    norm = np.sqrt(np.sum(np.square(vec)))
    return vec / norm
    

def main():
    seed = 5
    rs = np.random.RandomState(seed)
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = rs.random(X.shape)
    
    Z = gaussian_filter(Z, sigma=7)
    Z -= Z.min()
    Z /= Z.max()
    Zinter = interpolate.interp2d(x, y, Z, kind='cubic')
    
    xidx, yidx = np.unravel_index(rs.randint(np.prod(Z.shape)), Z.shape)
    gradpt1 = np.array([X[xidx, yidx], Y[xidx, yidx], Z[xidx, yidx]])
    dxdy = np.array([gpt[xidx, yidx] for gpt in np.gradient(Z, x, y)])
    dy, dx = normalize(dxdy) * 1e-4
    dz = Zinter(gradpt1[0] + dx, gradpt1[1] + dy)[0] - Z[xidx, yidx]
    grad = normalize(np.array([dx, dy, dz])) * 0.5
    
    fig = mlab.figure(bgcolor=(1,1,1))
    su = mlab.surf(X.T, Y.T, Z.T)
    sc = mlab.points3d([X[xidx, yidx]], [Y[xidx, yidx]], [Z[xidx, yidx]],
                       scale_factor=0.1, scale_mode='none',
                       opacity=1.0, resolution=20, color=(1,0,0))
    cmap_name = 'viridis'
    cdat = np.array(get_cmap(cmap_name,256).colors)
    cdat = (cdat*255).astype(int)
    su.module_manager.scalar_lut_manager.lut.table = cdat
    
    arrow = Arrow(gradpt1, gradpt1 + grad)
    
    mlab.view(azimuth=250, elevation=55)
    
    mlab.savefig('../images/gradient.png')
    
    mlab.show()
    return


if __name__ == "__main__":
    main()
