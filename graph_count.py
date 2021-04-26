import numpy as np
from pathlib import Path
from pyflagsercount import flagser_count
from typing import List, Dict

default_temporary_name = "temp"


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


def biedge_counts_per_dimension(conn_matrix: np.ndarray, repeats: bool = True,
                                temp_fname: str = default_temporary_name) -> Dict[int, int]:
    """
    Count of bidirectional edges per dimension of a given graph represented by a connectivity matrix.
    :param conn_matrix: (np.ndarray) connectivity matrix of the graph.
    :param repeats: (bool) whether to count bidirectional edges as they appear or to divide them based
        on the maximal dimension of the simplex they belong to.
    :param temp_fname: (str) temporary filename stem (i.e. no extension) to use. temp files will be
        "temp_fname%d.binary", where %d% is the thread number from flagser-count.
    :return result_dictionary: (Dict[int,int]) a dictionary containing the {dimension: count} pairs.
    """
    # Check temp files do not already exist
    for file in Path("").glob(temp_fname + "*.binary"):
        if file.exists():
            raise FileExistsError("File " + str(file.absolute()) + " already exists. Aborting.")

    # Retrive simplex list from flagser-generated files
    def get_simplex_list():
        simplices = []
        for file in Path("").glob(temp_fname + "*.binary"):
            simplices.extend(binary2simplex(file))
        return simplices

    # Need try/finally for cleanup
    try:
        # Calls flagser-count binding
        flagser_count(conn_matrix, binary=temp_fname, min_dim_print=1)

        if repeats:
            return count_biedges_with_repeats(conn_matrix, get_simplex_list())
        else:
            return count_biedges_without_repeats(conn_matrix, get_simplex_list())

    finally:
        # Remove files generated by flagser-count
        for file in Path("").glob(temp_fname + "*.binary"):
            file.unlink(missing_ok=True)


# Count bidirectional edges as found in simplices
def count_biedges_with_repeats(conn_matrix, simplices):
    result_dict = {}
    for simplex in simplices:
        dimension = len(simplex) - 1
        result_dict[dimension] = result_dict.setdefault(dimension, 0) + \
                                 biedges_count_in_simplex(conn_matrix, simplex)
    return result_dict


# Count bidirectional edges according to the maximum dimension of simplex they belong to
def count_biedges_without_repeats(conn_matrix, simplices):
    # This matrix stores the maximum dimension reached by an edge
    dimension_record_matrix = np.zeros(conn_matrix.shape, dtype=np.int)
    for simplex in simplices:
        dimension = len(simplex) - 1
        biedges_coordinates = biedges_coordinates_in_simplex(conn_matrix, simplex)
        for biedge in zip(*biedges_coordinates):
            # A bidirectional edge is represented by its sorted nodes
            sorted_biedge = tuple(sorted(biedge))
            if dimension_record_matrix[sorted_biedge] < dimension:
                dimension_record_matrix[sorted_biedge] = dimension
    dimensions, counts = np.unique(dimension_record_matrix, return_counts=True)
    result_dict = dict(zip(dimensions, counts))
    # Some zeros remain from both edges that are not bidirectional and triu part of matrix
    result_dict.pop(0, None)
    return result_dict


def biedges_count_in_simplex(conn_matrix: np.ndarray, simplex: List[int]):
    return np.count_nonzero(np.triu(conn_matrix[simplex].T[simplex]))


def biedges_coordinates_in_simplex(conn_matrix: np.ndarray, simplex: List[int]):
    biedges_indices_in_simplex = np.nonzero(np.triu(conn_matrix[simplex].T[simplex]))
    biedges_rows_in_matrix = [simplex[i] for i in biedges_indices_in_simplex[1]]
    biedges_cols_in_matrix = [simplex[i] for i in biedges_indices_in_simplex[0]]
    return biedges_rows_in_matrix, biedges_cols_in_matrix
