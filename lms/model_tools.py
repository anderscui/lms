# coding=utf-8
import math


def rope_theta(vec_dim: int, init_theta, p: int, pos: int):
    return 1/(math.pow(init_theta, 2*(p-1)/vec_dim)) * pos


if __name__ == '__main__':
    dim = 200
    theta = 500_000
    pos = 100
    for p in range(1, int(dim/2+1)):
        print(p, rope_theta(dim, theta, p, pos))
