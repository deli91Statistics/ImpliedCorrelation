import numpy as np
import pandas as pd
from numba import jit

from arch import arch_model


def compute_returns(df, return_type='simple', frequency='D', demean=True):
    """
    :param df: financial prices, assuming daily frequency
    :param type: net or log
    :param freq: Week W, Month M, Year Y
    :return:
    """
    
    assert (df < 0).any().sum() == 0 , "Negative price values"
    
    fin_df = df.copy()

    if frequency == 'W':
        fin_df = fin_df[::5]
    if frequency == 'M':
        fin_df = fin_df[::21]
    if frequency == 'Y':
        fin_df = fin_df[::252]

    if return_type == 'simple':
        fin_return = fin_df.pct_change()
        fin_return = fin_return.dropna()
    elif return_type == 'log':
        fin_return = np.log(fin_df)
        fin_return = fin_return.diff(1)
        fin_return = fin_return.dropna()
    else:
        raise Exception("Return type has to be either 'net' or 'log'")
    
    if demean:
        fin_return = (fin_return - fin_return.mean())/fin_return.std()

    return fin_return


def compute_tail_risk(data_array, risk_lvl, risk_measure, tail):
    
    assert isinstance(data_array, np.ndarray), 'Input is not a np.array'
    assert risk_measure in ['VaR', 'ES'], 'Wrong risk measure specified'
    
    if tail == 'lower':
        VaR = [np.quantile(data_array[:, i], alpha, method='lower') 
               for i, alpha in zip(range(data_array.shape[1]), risk_lvl)]
        if risk_measure == 'VaR':
            return np.array(VaR)
        elif risk_measure == 'ES':
            ES = [data_array[:,i][data_array[:,i] < VaR[i]].mean() for i in range(data_array.shape[1])]
            return np.array(ES)

    if tail == 'upper':
        VaR = [np.quantile(data_array[:, i], 1-alpha, method='higher') 
               for i, alpha in zip(range(data_array.shape[1]), risk_lvl)]
        if risk_measure == 'VaR':
            return np.array(VaR)
        elif risk_measure == 'ES':
            ES = [data_array[:,i][data_array[:,i] > VaR[i]].mean() for i in range(data_array.shape[1])]
            return np.array(ES)


def create_portfolios(asset_data, weight_matrix):
    """
    Simply attaching the weights to the dataset
    :param asset_data: (return) data
    :param weight_matrix: weights, here: generated by weights_identification
    :return: a set of portfolios
    """
    assert asset_data.shape[1] == weight_matrix.shape[1] , 'Dimension missmatch'
    return asset_data.dot(weight_matrix.T) 


def weights_one_asset(n_assets):
    return np.eye(n_assets)


def weights_two_assets_equal(n_assets):
    # Creates weight matrix: Equal weights, 2 assets, n_assets universe

    # 1 Initiate first weight matrix, first column contains 0.5 and then only diagonals
    # 2 Set up recursion start

    # Algorithmic idea:
    # 3 Cancel last column and row of the recurrent weight matrix
    # 4 Add one column of zeros at the beginning and stack new recurrent matrix horizontally
    # 5 Add the new weight matrix to the previous on
    # 6 Loop through range to save all permutations in one

    # Note :  Row is one less than columns due to possible pairs, thus idx starts at 2, recall index starts at 0

    weight_matrix_init = np.hstack([np.ones((n_assets-1, 1)),
                                    np.eye(n_assets-1)])*0.5  # 1

    weight_matrix = weight_matrix_init  # 2
    weight_matrix_recursive = weight_matrix

    for idx in range(2, n_assets):  # 6
        weight_matrix_recursive = weight_matrix_recursive[:n_assets - idx, :n_assets - 1]  # 3
        zero_column = np.zeros((n_assets - idx, 1))

        weight_matrix_recursive = np.hstack([zero_column, weight_matrix_recursive])  # 4

        weight_matrix = np.vstack([weight_matrix, weight_matrix_recursive])  # 5

    return weight_matrix


def weights_three_assets_equal(n_assets):
    # Generate weights for a three asset portfolio in an n_asset universe
    # No clue how the algorithm works...but it works...

    # Technical notes:
    # a range() does not include the bounds => add 1
    # b np.shape returns a tuple => call first element with tuple[0 or 1]

    # 1 Initiate one zero line for recursion start and delete at the end

    n_blocks = n_assets-2

    weight_matrix = np.zeros((1, n_assets))  # 1

    for idx in range(1, n_blocks+1):  # a
        w1 = np.kron(np.eye(idx), np.ones((n_blocks+1-idx, 1)))
        w2 = np.sum(w1, axis=1).reshape(np.shape(w1)[0], 1)  # b
        w3 = np.kron(np.ones((idx, 1)), np.eye(n_blocks+1-idx))
        weight_matrix = np.vstack([weight_matrix,
                                   np.hstack([w1, w2, w3])])
    return weight_matrix[1:, :]/3  # 1


def weights_all_minus_three_assets_equal(n_assets):
    # Compute the weights for portfolios with at least 5 assets
    # Otherwise it doesn't make sense

    # Algorithmic Idea: Reverse three asset portfolio
    #   Turn all 0 weights to 1
    #   Turn all not 1 values to 0
    #   Divide array by n_assets-3

    weight_matrix = weights_three_assets_equal(n_assets)
    weight_matrix[weight_matrix == 0] = 1
    weight_matrix[weight_matrix != 1] = 0
    return weight_matrix/(n_assets-3)


def weights_all_minus_two_assets_equal(n_assets):
    # Compute the weights for portfolios with at least 4 assets
    # Otherwise it doesn't make sense

    # Algorithmic Idea: Reverse two asset portfolio
    #   Turn all 0.5 weights to 0
    #   Turn all 0 to 1
    #   Divide array by n_assets-2

    weight_matrix = weights_two_assets_equal(n_assets)
    weight_matrix[weight_matrix == 0] = 1
    weight_matrix[weight_matrix == 0.5] = 0
    return weight_matrix/(n_assets-2)


def weights_all_minus_one_asset_equal(n_assets):
    # Create matrix of ones and subtract the identity matrix and divide by n_assets
    weight_matrix = np.ones((n_assets, n_assets))-np.eye(n_assets, n_assets)
    return weight_matrix/(n_assets-1)


def weights_all_assets_equal(n_assets):
    return np.ones(n_assets)/n_assets


def weights_exact_identification(n_assets):
    return weights_all_assets_equal(n_assets)


def weights_dirichlet_uniforn(n_assets):
    dirichlet_weights = tuple(np.ones(n_assets))
    n_portfolios = int((n_assets*(n_assets-1))/2)
    return np.random.dirichlet(dirichlet_weights, n_portfolios)


def weights_two_assets_random(n_assets):
    """
    Creates a weight matrix with random weights for two-asset portfolios,
    ensuring the sum of weights for each portfolio equals 1.
    
    Parameters:
    - n_assets (int): Number of assets in the universe.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - weight_matrix (ndarray): A matrix of random weights for two-asset portfolios.
    """
    
    weight_matrix = []

    # Generate random weights for all two-asset combinations
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            weight_1 = np.random.rand()  # Random weight for the first asset
            weight_2 = 1 - weight_1  # Remaining weight for the second asset
            
            # Create a portfolio weight row
            portfolio = np.zeros(n_assets)
            portfolio[i] = weight_1
            portfolio[j] = weight_2
            
            weight_matrix.append(portfolio)
    
    # Convert the list to a NumPy array
    weight_matrix = np.array(weight_matrix)
    return weight_matrix

def weights_two_assets_random_with_negatives(n_assets):
    """
    Creates a weight matrix with random weights (including negatives) 
    for two-asset portfolios, ensuring the sum of weights for each portfolio equals 1.
    
    Parameters:
    - n_assets (int): Number of assets in the universe.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - weight_matrix (ndarray): A matrix of random weights for two-asset portfolios.
    """

    weight_matrix = []

    # Generate random weights for all two-asset combinations
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            weight_1 = np.random.uniform(-1, 1)  # Random weight for the first asset
            weight_2 = 1 - weight_1  # Remaining weight for the second asset
            
            # Create a portfolio weight row
            portfolio = np.zeros(n_assets)
            portfolio[i] = weight_1
            portfolio[j] = weight_2
            
            weight_matrix.append(portfolio)
    
    # Convert the list to a NumPy array
    weight_matrix = np.array(weight_matrix)
    return weight_matrix





def weights_identification(n_assets, weight_case):
    # 1 Exact identification only given in the two asset portfolio case
    #   The number of possible 2 asset portfolio is then equal to the number of correlation pairs
    #   thus: # equations = # correlation pairs (correlation coefficients)
    # 2 All other cases refer to overidentification
    # Interpretation of the output: One row correspond to one weight vector of a portfolio
    # row = portfolio, column = asset

    if weight_case == 1:
        return weights_two_assets_equal(n_assets)  # 1
    elif weight_case == 2:  # 2
        weight_2 = weights_two_assets_equal(n_assets)
        weight_3 = weights_three_assets_equal(n_assets)
        weights_2_3 = np.vstack([weight_2, weight_3])
        return weights_2_3
    elif weight_case == 3:
        weight_2 = weights_two_assets_equal(n_assets)
        weight_n3 = weights_all_minus_three_assets_equal(n_assets)
        weights_2_n3 = np.vstack([weight_2, weight_n3])
        return weights_2_n3
    elif weight_case == 4:
        weight_2 = weights_two_assets_equal(n_assets)
        weights_n2 = weights_all_minus_three_assets_equal(n_assets)
        weights_2_n2 = np.vstack([weight_2, weights_n2])
        return weights_2_n2
    elif weight_case == 5:
        w = weights_dirichlet_uniforn(n_assets)   # exact case only
        return w
    elif weight_case == 6:
        w = weights_two_assets_random(n_assets)  # exact case only
        return w
    elif weight_case == 7:
        w = weights_two_assets_random_with_negatives(n_assets)  # exact case only
        return w
    else:
        raise Exception('Identification wrong')


def filter_garch_effects(dataframe, p_order=1, q_order=1, err_dist='normal'):
    """
    Filter a dataframe for heteroscedasticity effects by using a GARCH model, one garch model per column
    :param dataframe: either a pandas Series or pandas dataframe
    :param p_order: autoregressive order
    :param q_order: lag of variance
    :param err_dist: error distribution
    :param demean: centralized data
    :return: dataframe with garch filtered data, dictionary with fitted models
    """
    if type(dataframe) == pd.Series:
        dataframe = dataframe.to_frame()

    fitted_garch_models = {}
    garch_filtered_data_lst = []

    for col in dataframe.columns:
        garch_model = arch_model(dataframe[col], p=p_order, q=q_order, dist=err_dist, rescale=False)
        garch_model_fit = garch_model.fit(update_freq=0, disp='off')
        filtered_data = garch_model_fit.resid / garch_model_fit.conditional_volatility
        fitted_garch_models[col] = garch_model_fit
        garch_filtered_data_lst.append(filtered_data)

    garch_filtered_data = pd.DataFrame(np.transpose(garch_filtered_data_lst), index=dataframe.index,
                                       columns=dataframe.columns)

    return garch_filtered_data, fitted_garch_models



# jit kann keine dicts
# nur recursion als low lvl funktion
# dachfunktion

def introduce_garch_effects(garch_model, residuals, burn_in=30):
    """
    Simulate a GARCH(1,1) process with predetermined innovations, e.g. from bootstrap, simulated residuals
    :param garch_params: Contains the list of parameters
    :param residuals: list, array of residuals
    :param init_x: initial value of the data
    :param init_sigma_square: initial value of the variance
    :return: numpy array with the simulated time series
    """
    n = len(residuals)

    alpha_0 = garch_model['omega']
    alpha_1 = garch_model['alpha[1]']
    beta_1 = garch_model['beta[1]']
    x = np.zeros(n)
    u = residuals
    sigma2 = np.zeros(n)

    y = garch11_recursion(alpha_0, alpha_1, beta_1, x, u, sigma2)

    # Brauche ich Anfangswerte? Jein, wird korrigiert durch burn-in phase
    # x[0] = 0
    # sigma2[0] = 0

    # for i in range(1, n):
    #     sigma2[i] = alpha_0 + alpha_1 * (x[i - 1] ** 2) + beta_1 * sigma2[i - 1]
    #     x[i] = np.sqrt(sigma2[i]) * u[i]

    # ggfs. x[500:,:] falls mit burn-in phase
    return y[burn_in:]


@jit(nopython=True)
def garch11_recursion(alpha_0, alpha_1, beta_1, x, u, sigma2):
    x[0] = 0
    sigma2[0] = 0

    for i in range(1, len(u)):
        sigma2[i] = alpha_0 + alpha_1 * (x[i - 1] ** 2) + beta_1 * sigma2[i - 1]
        x[i] = np.sqrt(sigma2[i]) * u[i]
        
    return x
    
    
    
    