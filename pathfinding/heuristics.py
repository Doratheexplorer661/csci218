"""Lower bounds for unit cardinal moves and sqrt(2) diagonal moves."""
from math import hypot, sqrt


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def diagonal(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)


HEURISTICS = {'manhattan': manhattan, 'euclidean': euclidean, 'diagonal': diagonal}
