import numpy as np
import pickle
from os import path


def find_locations(str, substr):
    start = 0
    while True:
        start = str.find(substr, start)
        if start == -1: return
        yield start
        start += len(substr)


def a2g(x):
    return (n for n in x)


def random_rows(A, num_rows, replace=False):
    return A[np.random.choice(A.shape[0], num_rows, replace=replace), :]


def save(stuff, filename, ext='pkl', folder='data'):
    with open(path.join(folder, '{}.{}'.format(filename, ext)), 'wb') as file:
        pickle.dump(stuff, file)


def load(filename, ext='pkl', folder='data'):
    with open(path.join(folder, '{}.{}'.format(filename, ext)), 'rb') as file:
        return pickle.load(file)


def load_or_create(filename, ext='pkl', create_fn=None, folder='data', with_status=False):
    try:
        magic_obj = load(filename, ext, folder)
        status = True
    except FileNotFoundError:
        status = False
        if create_fn is None:
            magic_obj = None
        else:
            magic_obj = create_fn()
            save(magic_obj, filename, ext, folder)
    if with_status:
        return status, magic_obj
    else:
        return magic_obj


def tuple_array_to_ndarray(tuple_array):
    return tuple_array_transpose(tuple_array)


def ndarray_to_tuple_array(ndarray):
    return tuple_array_transpose(ndarray)


def tuple_array_transpose(m):
    return list(zip(*m))
