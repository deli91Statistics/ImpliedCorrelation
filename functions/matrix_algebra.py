import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.linalg import khatri_rao

from scipy.stats import uniform



def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def vec(ndarray):
    """
    Vectorize a matrix
    :param ndarray: matrix, ndarray
    :return: A vector
    """
    return ndarray.flatten('F')


def vecl(ndarray_matrix):
    """
    Vectorize a matrix by stacking the columns below the main diagonal.
    Note: np.tril / np.triu yields the indices of the entries by fixing a row and moving through the columns. Using this only
          yields vectorization by row, i.e. moves through the matrix in the following pattern (np.tril)
          
          1
          2 3
          4 5 6
          7 8 9 10 
          
          Instead we want
          
          1
          2 5 
          3 6 8
          4 7 9 10
    
    For our purpose this is solved by using the upper triangular matrix and transpose in the end
    :param ndarray_matrix: ndarray: matrix, ndarray
    :return: lower vectorized matrix/array
    """
    if not check_symmetric(ndarray_matrix):
        raise Exception('Matrix not symmetric')

    lower_matrix_indices = np.triu_indices(ndarray_matrix.shape[0], 1)
    return ndarray_matrix.T[lower_matrix_indices]


def antivecl(ndarray_vector):
    # 1 Fix the length of vector
    # 2 Formula derived from solution for quadratic equations: dim_mat = dim_vec*(dim_vec-1)/2, see vecl
    # 3 Check if length matches the formula, of a float is given, then length of vector is wrong
    # 4 Initialize empty matrix
    # 5 Fill lower/upper/diagonal matrix, np.tril_indices returns index of lower triangle by row
    # Note: Numpy fills by row and not by column, in theoretical work, the vecl operator stacks by column
    #       therefore we have to work with triu and not tril in #5

    dim_vec = len(ndarray_vector)  # 1
    dim_mat = int(1 / 2 + np.sqrt(1 / 4 + 2 * dim_vec))  # 2
    assert type(dim_mat) == type(dim_vec)  # 3

    matrix = np.zeros((dim_mat, dim_mat))  # 4

    matrix[np.triu_indices(dim_mat, +1)] = ndarray_vector  # 5
    # matrix[np.tril_indices(dim_mat, -1)] = ndarray_vector  # 5
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 1)
    return matrix


def antivecl_lower(ndarray_vector):
    # 1 Fix the length of vector
    # 2 Formula derived from solution for quadratic equations: dim_mat = dim_vec*(dim_vec-1)/2, see vecl
    # 3 Check if length matches the formula, of a float is given, then length of vector is wrong
    # 4 Initialize empty matrix
    # 5 Fill lower/upper/diagonal matrix, np.tril_indices returns index of lower triangle by row
    # Note: Numpy fills by row and not by column, in theoretical work, the vecl operator stacks by column
    #       therefore we have to work with triu and not tril in #5

    dim_vec = len(ndarray_vector)  # 1
    dim_mat = int(1 / 2 + np.sqrt(1 / 4 + 2 * dim_vec))  # 2
    assert type(dim_mat) == type(dim_vec)  # 3

    matrix = np.zeros((dim_mat, dim_mat))  # 4

    matrix[np.triu_indices(dim_mat, +1)] = ndarray_vector  # 5
    # matrix[np.tril_indices(dim_mat, -1)] = ndarray_vector  # 5
    return matrix


def unit_vector_matlab(dim, idx):
    # Return a unit vector, i.e. i-th column of identity matrix
    # Create zero vector with correct dimension and change (i-1)-th entry to 1 since index starts at 0
    assert dim != 0
    assert idx != 0
    unit_vec = np.zeros([dim, 1])
    unit_vec[idx - 1] = 1
    return unit_vec


def duplicate_vecl(dim):
    """
    Generates duplication matrix D such that

            vec(R) = Dvecl(R)

    where R is a symmetric n x n (correlation) matrix and vecl(R) contains
    the elements of R below the main diagonal. The (missing) diagonal
    elements in vec(R) are replaced by zeros.

    I have no clue how this algorithm works, see mittnik code duplicatvecl.m, but it works
    :param dim:
    :return:
    """
    id = np.eye(dim)

    matrix_iteration = np.zeros((dim * dim, 1))

    for idx_1 in range(1, dim):
        dd = np.vstack(
            [
                np.zeros(((idx_1 - 1) * dim, dim - idx_1 + 1)),
                id[:, idx_1 - 1:]
            ]
        )
        for idx_11 in range(1, dim - idx_1 + 1):
            dd = np.vstack(
                [
                    dd,
                    unit_vector_matlab(dim, idx_1).dot(unit_vector_matlab(dim - idx_1 + 1, 1 + idx_11).T)
                ]
            )
        matrix_iteration = np.hstack([matrix_iteration, dd[:, 1:]])

    return matrix_iteration[:, 1:]


def generalized_kronecker(matrix_1, matrix_2, ax='row'):
    """
    row: row-wise kronecker product
    col: column-wise kronecker product also know as Khatri-Rao
    """
    if ax == 'row':
        X = khatri_rao(matrix_1.transpose(), matrix_2.transpose())
        X = X.transpose()

    if ax == 'col':
        X = khatri_rao(matrix_1, matrix_2)

    return X


def unit_vec_idx(col_counter, n_col):
    """
    Returns row index of entry for the selection matrix, which is used to convert a khatri-rao product to kronecker product. See Lev-Ari (2005)
    Subtract 1 because python arrays start at index 0
    :param col_counter: counts through each columns
    :param n_col: Number of columns
    :return:
    """
    return (col_counter * n_col) + (col_counter + 1) - 1


def khatri_kronecker_selection_matrix(n_col, make_sparse=False):
    """
    See: Lev-Ari (2005):
    Creates a matrix S such that: A [khatri-rao] B = (A [kron] B) @ S
    where the khatri-rao product is the COLUMN-wise Kronecker product.
    
    Args:
        n_col (int): number of columns

    Returns:
        nparray: selection matrix as described in the paper
        
    # Example 1: Khatri-Rao -> Kronecker
        A = np.array([[14, 2, 1, 5], [2, 1, 0, 1], [2, 2, 5, 2]])
        B = np.array([[14, 4, 1, 6], [0, 7, 22, 3], [2, 9, 1, 4], [9, 1, 3, 10]])
        X = generalized_kronecker(A, B, ax='col')           # Khatri-Rao product
        Y = np.kron(A, B)
        S = khatri_kronecker_selection_matrix(4)
        X == Y @ S
    """
    row_idx = np.array([unit_vec_idx(col, n_col) for col in range(n_col)])
    col_idx = np.array(range(n_col))
    
    if make_sparse:
        S_coo = coo_matrix((np.ones(n_col),(row_idx,col_idx)))
        S_csc = csc_matrix(S_coo)
        return S_csc
    else:
        S = np.zeros((n_col**2, n_col))
        S[([row_idx],[col_idx])] = 1
        return S


def compute_D_tilde(no_assets, no_portfolio):
    """
    Used to deriving the derivatives. Used to rearrange vec(dq' [kron] I_m) in order to isolated dq
    See personal notes: FIN 3 (13) bottom
    Best used in conjunction with weight_identification for a dynamic assignment of #assets and #portfolio
    """
    D_tilde = np.zeros((no_portfolio ** 2 * no_assets, no_assets))
    vec_Id = np.identity(no_portfolio).flatten('F')

    for i in range(no_assets):
        D_tilde[i * no_portfolio ** 2:(i + 1) * no_portfolio ** 2, i] = vec_Id
    return D_tilde


# Example: Generic
# weights = weights_identification(4, 1)
# n_assets = weights.shape[1]
# n_portfolio = weights.shape[0]
# D_tilde = rearrange_kronecker_identity(n_assets, n_portfolio)

# Example: with numbers
# row = np.array([[1, 2, 3, 4]])
# col = np.array([[1, 2, 3, 4]]).T
# id = np.identity(6)
# kron = np.kron(row, id)
# vec_kron = list(vec(kron))
# vec_kron = np.array([vec_kron]).T
# D_tilde = rearrange_kronecker_identity(4, 6)
# comp = D_tilde @ col
# comp == vec_kron


def duplicate_vector_alternating(v: list, row_iter, col_iter):
    """
    Given a column vector, duplicate vector m-times along the row and the entire block
    m x n times vertically; this is the first component needed to compute a kronecker product.
    See FIN 4 (15*). The matrix will be later multiplied with C component-wise
    :param v: in research context: v = vec(Z_alpha)
    :param col_iter: number of rows
    :param row_iter: number of create columns
    :return:
    """
    column_vector = np.array([v]).T
    return np.tile(column_vector, (row_iter, col_iter))

# Example 1, Idea:
# n = 4
# lst = [1, 2, 3, 4]
# col = np.array([lst]).T
# A = np.tile(col, (n*col.shape[0], n))

# Example 2, Comparison:
# lst = [1, 2, 3, 4]
# dupl_vec = duplicate_vector_alternating(lst, 2, 4)


def duplicate_matrix_rows_blockwise(matrix_array, iterations):
    """
    Duplicate each row of a matrix blockwise and stack them in the end.
    This is the second component needed to compute a kronecker product.
    Also used to create a blockwise vector.
    See FIN 4 (15*) and (15**) and (4*)
    """
    lst = []
    for row in matrix_array:
        lst.append(np.tile(row, (iterations, 1)))
    lst_final = np.concatenate(lst, axis=0)
    return lst_final

# Example 1: Blockwise matrix
# A = np.array([[14, 2, 1, 5], [2, 1, 0, 1], [2, 2, 5, 2]])
# duplicate_matrix_rows_blockwise(A, 3)


# Example 2: Blockwise COLUMN vector
# A = np.array([[14, 2, 1, 5]]).T
# duplicate_matrix_rows_blockwise(A, 3)


def duplicate_matrix_alternating(matrix_array, iterations):
    """
    Given a matrix, replicate the matrix [iteration]-times and stack them above each other
    Needed to compute a kronecker product.
    See FIN 4 (15**)
    :param matrix_array:
    :param iterations:
    :return:
    """
    return np.tile(matrix_array, (iterations, 1))

# Example
# M = np.array([[14, 2, 1, 5], [2, 1, 0, 1], [2, 2, 5, 2]])
# duplicate_matrix_alternating(M, 4)

# Example
# v = np.array([[14], [2], [2]])
# duplicate_matrix_alternating(v, 4)

#################################################################
# Truncation & Control for positive semi-definite matrix
#################################################################

# note: most time spent here according to profiler

def is_positive_semi_definite_matrix(matrix):
    eig_val, eig_vec = np.linalg.eig(matrix)
    check = eig_val.min() >= 0
    return check


def truncate_correlation_matrix(corrmat):
    corrmat[corrmat > 1] = 1
    corrmat[corrmat < -1] = -1
    return corrmat


def set_negative_EV_to_zero(matrix):
    eig_val, eig_vec = np.linalg.eig(matrix)
    eig_val[eig_val < 0] = 0.00005
    return eig_val, eig_vec


def set_negative_EV_to_random(matrix, lower_bound, upper_bound):
    eig_val, eig_vec = np.linalg.eig(matrix)
    bad_eig_counter = len(eig_val[eig_val < 0])
    eig_val[eig_val < 0] = np.random.uniform(lower_bound, upper_bound, bad_eig_counter)  # 3
    return eig_val, eig_vec


def rescale_matrix(matrix):
    scale_mat = np.zeros(matrix.shape)
    diag_val = 1/np.diag(matrix)
    np.fill_diagonal(scale_mat, diag_val)
    rescaled_mat = scale_mat @ matrix
    return rescaled_mat


def check_symmetric_matrix(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
