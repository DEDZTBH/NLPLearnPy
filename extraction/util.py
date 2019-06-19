import numpy as np


def find_locations(str, substr):
    start = 0
    while True:
        start = str.find(substr, start)
        if start == -1: return
        yield start
        start += len(substr)


def random_rows(A, num_rows, replace=False):
    return A[np.random.choice(A.shape[0], num_rows, replace=replace), :]
