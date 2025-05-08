import numpy as np
import pandas as pd
import functions.matrix_algebra as malg



def correlation_bounds_correction(correlation_matrix_estimate):
    """
    Set all entries in a correlation matrix that exceeds +1 or -1
    to their respective bounds.
    :param correlation_matrix_estimate: Estimated correlation matrix
    :return: returns a matrix with truncated boundaries
    """
    matrix_truncated = correlation_matrix_estimate.copy()
    matrix_truncated[matrix_truncated > 1] = 1
    matrix_truncated[matrix_truncated < -1] = -1
    return matrix_truncated


def spectral_correction(correlation_matrix_estimate, method='random', bounds=[0.000001, 0.000005]):
    """
    Check if correlation matrix positive semi-definite. If not, correct with spectral decomposition
    :param bounds: upper and lower bounds
    :param method: zero sets negative EV to zero, random sets negative EV to some RN~unif(bounds)
    :param correlation_matrix_estimate: Estimated Correlation matrix
    :return: returns a fixed matrix with logs
    Reference:
    Rebonato_Jaeckel(1999) & http://www.deltaquants.com/manipulating-correlation-matrices

    NOTE: Setting the method to zeros yields a non positive semi-definite matrix, due to numerical issues...

    """
    fixed_matrix = correlation_matrix_estimate.copy()
    if method == 'zero':
        fixed_EV, eig_vec = malg.set_negative_EV_to_zero(fixed_matrix)
    elif method == 'random':
        fixed_EV, eig_vec = malg.set_negative_EV_to_random(fixed_matrix, bounds[0], bounds[1])
    else:
        raise ValueError('No correction method specified')

    scaling_mat = np.sqrt(np.diag(1/(np.square(eig_vec) @ fixed_EV)))
    sqrt_EV_mat = np.sqrt(np.diag(fixed_EV))
    B = scaling_mat @ eig_vec @ sqrt_EV_mat
    fixed_matrix = B @ B.T

    return fixed_matrix


def generate_corr_pair_names_colwise(n, names):    
    ro, co = np.triu_indices(n, 1)
    return [f'{names[name_1]} : {names[name_2]}' for name_1, name_2 in zip(ro,co)]
    
    
def get_correlation_pair_names(asset_names: np.array):

    # Creates a matrix were the entries are names of the column/row pairs
    # Needed for labeling correlation coefficients

    # 0 Initiate a empty list
    # 1 Double loop through the names to compute all possible permutations
    # 2 String operation: join the names with an space between the names
    # 3 Fill the list with the words/strings via append
    # 4 Rearrange the list into matrix form

    n_cols = asset_names.shape[0]

    corr_pairs_vector = []  # 0

    for idx_1 in range(len(asset_names)):  # 1
        for idx_2 in range(len(asset_names)):  # 1
            name_pair = " vs ".join([asset_names[idx_1], asset_names[idx_2]])  # 2
            # print(name_pair)
            corr_pairs_vector.append(name_pair)  # 3

    corr_pairs_matrix = np.reshape(corr_pairs_vector, (n_cols, n_cols)).transpose()  # 4

    return corr_pairs_matrix


def fisher_transform(rho_pearson):
    return 0.5*np.log((1+rho_pearson)/(1-rho_pearson))


def inverse_fisher_transform(rho_fisher):
    
    num = np.exp(2*rho_fisher) - 1
    denom = np.exp(2*rho_fisher) + 1
    return num/denom


def fill_matrix_with_corr_and_SE(asset_returns, corr_est, corr_se):

    n_assets = asset_returns.shape[1]
    tickers = asset_returns.columns

    df = pd.DataFrame(np.zeros((n_assets, n_assets)) + np.identity(n_assets), 
                                columns = tickers, 
                                index= tickers)

    entries = [str(round(qic, 3)) + ' ' + '(' + str(round(qic_se, 3))+ ')' for qic, qic_se in zip(corr_est, corr_se)]

    ro, co = np.triu_indices(n_assets, 1)

    i = 0
    for r,c in zip(ro,co):
        df.iloc[r,c] = entries[i]
        i += 1
    
    df.replace(0.0, '', inplace=True)
    
    return df.T


def merge_upper_lower_matrix(lower_matrix: pd.DataFrame, upper_matrix: pd.DataFrame, dtype = float) -> pd.DataFrame:
    """
    Merge upper and lower matrix dataframe into one dataframe

    Args:
        lower_matrix (pd.DataFrame): _description_
        upper_matrix (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: Merged Dataframe
    
    Example:
    
    data_A = {
        'A': [1, 0.8, 0.5, 0.2],
        'B': [0.4, 1, 0.6, 0.3],
        'C': [0.7, 0.9, 1, 0.4],
        'D': [0.2, 0.3, 0.5, 1]
        }
        
    data_B = {
        'A': [1, 0.1, 0.3, 0.4],
        'B': [0.5, 1, 0.2, 0.6],
        'C': [0.8, 0.3, 1, 0.7],
        'D': [0.9, 0.4, 0.6, 1]
        }
        
        df_A = pd.DataFrame(data_A, index=['A', 'B', 'C', 'D'])
        df_B = pd.DataFrame(data_B, index=['A', 'B', 'C', 'D'])

        merge_upper_lower_matrix(df_A, df_B)
    
    """
    
    merged_matrix = pd.DataFrame(index=lower_matrix.index, columns=lower_matrix.columns, dtype=dtype)
    
    d = lower_matrix.shape[0]
    
    lower_mask = np.tril_indices(d, k=-1)
    lower_vals = lower_matrix.values[lower_mask]

    upper_mask = np.triu_indices(d, k=1)
    upper_vals = upper_matrix.values[upper_mask]
    
    merged_matrix.values[lower_mask] = lower_vals
    merged_matrix.values[upper_mask] = upper_vals
    
    np.fill_diagonal(merged_matrix.values, 1)
    
    return merged_matrix


def remove_separator(number):
    """
    Used to convert a float, e.g., 0.5 to 05 as a string for naming the latex tables
    """

    str_number = str(number)
    str_number_no_separator = str_number.replace('.', '')
    
    return str_number_no_separator