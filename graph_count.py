from pyflagsercount import flagser_count
from typing import List
import numpy as np
from pathlib import Path

temporary_file_name = "temp"


def binary2simplex(address):
    """From Jason's flagser-count."""
    X = np.fromfile(address, dtype='uint64')                         #Load binary file
    S=[]                                                             #Initialise empty list for simplices

    i=0
    while i < len(X):
        b = format(X[i], '064b')                                     #Load the 64bit integer as a binary string
        if b[0] == '0':                                              #If the first bit is 0 this is the start of a new simplex
            S.append([])
        t=[int(b[-21:],2), int(b[-42:-21],2), int(b[-63:-42],2)]     #Compute the 21bit ints stored in this 64bit int
        for j in t:
            if j != 2097151:                                         #If an int is 2^21 this means we have reached the end of the simplex, so don't add it
                S[-1].append(j)
        i+=1
    return S


def biedge_count_per_dimension(conn_matrix: np.ndarray, repeats: bool = True):
    result_dict = {}

    for file in Path("").glob(temporary_file_name + "*.binary"):
        if file.exists():
            raise FileExistsError("File " + str(file.absolute()) + " already exists. Aborting.")

    flagser_count(conn_matrix, binary=temporary_file_name, min_dim_print=1)
    try:
        if repeats:
            for file in Path("").glob(temporary_file_name + "*.binary"):
                simplices = binary2simplex(file)
                for simplex in simplices:
                    dimension = len(simplex) - 1
                    result_dict[dimension] = result_dict.setdefault(dimension, 0) +\
                                             biedge_in_simplex(conn_matrix, simplex)
        else:
            dimension_record_matrix = np.zeros(conn_matrix.shape, dtype=np.int)

            simplices = []
            for file in Path("").glob(temporary_file_name + "*.binary"):
                simplices.extend(binary2simplex(file))
            for simplex in simplices:
                dimension = len(simplex) - 1
                biedge_coordinates = biedges_in_simplex_coordinates(conn_matrix, simplex)
                for biedge in zip(*biedge_coordinates):
                    sorted_biedge = tuple(sorted(biedge))
                    if dimension_record_matrix[sorted_biedge] < dimension:
                        dimension_record_matrix[sorted_biedge] = dimension
            dimensions, counts = np.unique(dimension_record_matrix, return_counts=True)
            result_dict = dict(zip(dimensions, counts))
        return result_dict
    finally:
        for file in Path("").glob(temporary_file_name + "*.binary"):
            file.unlink()


def biedge_in_simplex(conn_matrix: np.ndarray, simplex: List[int]):
    return np.sum(np.triu(conn_matrix[simplex].T[simplex]))


def biedges_in_simplex_coordinates(conn_matrix: np.ndarray, simplex: List[int]):
    biedges_indices_in_simplex = np.nonzero(np.triu(conn_matrix[simplex].T[simplex]))
    biedges_rows_in_matrix = [simplex[i] for i in biedges_indices_in_simplex[1]]
    biedges_cols_in_matrix = [simplex[i] for i in biedges_indices_in_simplex[0]]
    return biedges_rows_in_matrix, biedges_cols_in_matrix