import numpy as np
from argparse import ArgumentParser


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
    x, y = circle_points(8, r=0.4)
    xc, yc = circle_points(1000, r=0.4)
    
    t = np.pi / 2
    circ_path = change(xc, yc, t, Ap=1, Ac=0)
    circ_points = change(x, y, t, Ap=1, Ac=0)
    
    lines = []
    line = '\\draw[gray, dashed, tdplot_rotated_coords] '
    for xpt, ypt in np.array(circ_path).T:
        line += f'({xpt}, {ypt}) -- '
    line += 'cycle;'
    lines.append(line)
    
    for xpt, ypt in np.array(circ_points).T:
        line = (f'\\draw[fill=black, tdplot_rotated_coords] ({xpt}, {ypt}, 0) '
                'circle (0.5pt);')
        lines.append(line)
    
    with open('../tikz/gw_points.tex', 'w') as fp:
        for line in lines:
            fp.write(line + '\n')


if __name__ == "__main__":
    main()
