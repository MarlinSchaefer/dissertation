import numpy as np
from argparse import ArgumentParser


def RotationMatrix(angle, axis='z'):
    def get_axis(axis):
        if isinstance(axis, str):
            if axis == 'x':
                axis = np.array([1, 0, 0])
            elif axis == 'y':
                axis = np.array([0, 1, 0])
            elif axis == 'z':
                axis = np.array([0, 0, 1])
        return normalize_axis(axis)
    
    def normalize_axis(axis):
        return axis / np.sqrt(np.sum(np.square(axis)))
    
    x, y, z = get_axis(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    matr = [[c + x**2 * (1 - c),
             x * y * (1 - c) - z * s,
             x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s,
             c + y**2 * (1 - c),
             y * z * (1 - c) - x * s],
            [x * z * (1 - c) - y * s,
             y * z * (1 - c) + x * s,
             c + z**2 * (1 - c)]]
    return np.array(matr)


def EulerRotation(alpha, beta, gamma, order='zyz'):
    angles = [alpha, beta, gamma]
    mat = np.eye(3)
    for angle, axis in zip(angles, order):
        mat = np.matmul(mat, RotationMatrix(angle, axis=axis))
    return mat


class EulerAngles(object):
    def __init__(self, matrix):
        self.matrix = matrix
    
    def convert_zyz(self):
        if self.matrix[2, 2] < 1:
            if self.matrix[2, 2] > -1:
                thetaY = np.arccos(self.matrix[2, 2])
                thetaZ0 = np.arctan2(self.matrix[1, 2], self.matrix[0, 2])
                thetaZ1 = np.arctan2(self.matrix[2, 1], -self.matrix[2, 0])
            else:
                thetaY = np.pi
                thetaZ0 = np.arctan2(self.matrix[1, 0], self.matrix[1, 1])
                thetaZ1 = 0
        else:
            thetaY = 0
            thetaZ0 = np.arctan2(self.matrix[1, 0], self.matrix[1, 1])
            thetaZ1 = 0
        return thetaZ0, thetaY, thetaZ1


def main():
    parser = ArgumentParser()
    
    parser.add_argument('angles', type=float, nargs='+',
                        help="The angles to use for the euler rotation.")
    parser.add_argument('--rotation-order', type=str, default='zyz',
                        help="The order of the axis for the euler rotations.")
    parser.add_argument('--get-euler-angles', action='store_true',
                        help="Return the total euler angles instead of the "
                             "total rotation matrix. (zyz)")
    parser.add_argument('--degree', action='store_true',
                        help="Set this flag if the input is in degrees.")
    
    args = parser.parse_args()
    
    if args.degree:
        args.angles = [angle / 180 * np.pi for angle in args.angles]
    
    if len(args.angles) > len(args.rotation_order):
        if len(args.rotation_order) % 3 == 0:
            n = int(np.ceil((len(args.angles) - len(args.rotation_order)) / 3))
            args.rotation_order = args.rotation_order + n * args.rotation_order[-3:]
    if len(args.angles) < len(args.rotation_order):
        args.angles.extend([0] * (len(args.rotation_order) - args.angles))
    
    mat = np.eye(3)
    for angle, axis in zip(args.angles, args.rotation_order):
        mat = np.matmul(mat, RotationMatrix(angle, axis=axis))
    
    if not args.get_euler_angles:
        print(mat)
        return
    
    euler_angles = EulerAngles(mat).convert_zyz()
    if args.degree:
        print(tuple(angle / np.pi * 180 for angle in euler_angles))
        return
    print(euler_angles)


if __name__ == "__main__":
    main()
