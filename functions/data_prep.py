import pandas as pd
import numpy as np
from typing import Union


def prep_raw_data_SX7P() -> pd.DataFrame:
    """
    1 Rename columns by row 3
    2 Drop first 5 rows (NaNs and useless excel inputs)
    3 Check and summarize columns with large number of NaN, get index of columns with no NaNs (see 3., 4., 5.)
    4 Adjust column names
    :return:
    """
    df_price = pd.read_excel('data/data_SX7P.xlsx', sheet_name='Price Data')
    df_price = df_price.copy()

    df_price.columns = df_price.iloc[2]  # 1
    df_price = df_price[5:]  # 2

    df_price['Dates'] = pd.to_datetime(df_price['Dates'], format='%d.%m.%Y')
    df_price.set_index('Dates', inplace=True)

    nan_counter = df_price.isna().sum()  # 3
    idx_non_nans = [idx for idx, val in enumerate(nan_counter) if val == 0]
    df_price = df_price.iloc[:, idx_non_nans]

    df_price = df_price.astype(float)

    col_names = np.array(df_price.columns)  # 4
    df_price.columns = [names.replace(' Equity', '') for names in col_names]
    return df_price


def prep_raw_data_DAX(offset=0, clear_NaN = False) -> pd.DataFrame:
    """
    The cleaned dataframe is stored as a csv file for later purposes. This stage
    performs a raw cleaning of the most fundamental dataflaws such as holidays,
    missing values, etc.
    """
    
    df_raw = pd.read_excel('data/raw/data_DAX_SP500.xlsx', sheet_name='DAX')
    df_price = pd.DataFrame(df_raw.iloc[5 + offset:,1:].values, 
                            columns=df_raw.iloc[3, 1:].values,
                            index=pd.to_datetime(df_raw.iloc[5 + offset :,0].values, format='%d.%m.%Y'))
    
    df_price.index.name = 'date'
    
    df_price.drop_duplicates(inplace=True)      # clear holidays
    
    if clear_NaN:
        nan_counter = df_price.isna().sum()
        idx_non_nans = [idx for idx, val in enumerate(nan_counter) if val == 0]
        df_price = df_price.iloc[:, idx_non_nans]
        
    df_price = df_price.astype(float)
    # df_price.iloc[df_price.index !=(0,1,2), :]            # all except...
    
    # store as csv...
    
    return df_price


def prep_global():
    df_raw = pd.read_excel('data/raw/data_global.xlsx', 
                           sheet_name = 'index', 
                           parse_dates=['Date'],
                           index_col = 0)
    
    df_raw.drop_duplicates(inplace=True)      # clear holidays
    
    return df_raw

#%%

def aggregate_simple_returns(daily_returns: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute the aggregated simple returns of a certain period.
    Used for computing weekly, monthly and yearly returns
    given daily(!) net returns.

    Args:
        daily_returns (Union[pd.Series, pd.DataFrame]): Daily net returns as a pandas Series or DataFrame.

    Returns:
        Union[pd.Series, pd.DataFrame]: Aggregated net returns as a pandas Series or DataFrame.
    """

    gross_returns = [1 + r for r in daily_returns]
    agg_gross_returns = np.prod(gross_returns)
    agg_net_return = agg_gross_returns - 1
    
    return agg_net_return


def aggregate_logarithmic_returns(log_daily_returns: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute the aggregated logarithmic returns of a certain period.
    Used for computing weekly, monthly, and yearly returns
    given daily(!) logarithmic returns.

    Args:
        log_daily_returns (Union[pd.Series, pd.DataFrame]): Daily logarithmic returns as a pandas Series or DataFrame.

    Returns:
        Union[pd.Series, pd.DataFrame]: Aggregated logarithmic returns as a pandas Series or DataFrame.
    """

    return np.sum(log_daily_returns)


def calculate_returns(data: Union[pd.DataFrame, pd.Series], 
                      frequency: str = 'D', 
                      log_returns: bool = False,
                      demean = False,
                      drop_na = True,
                      drop_zeros = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculates simple or logarithmic daily, weekly, monthly, yearly returns based on a Pandas DataFrame containing price data.

    Args:
        data (Union[pd.DataFrame, pd.Series]): Price data
        frequency (str): The frequency of the returns calculation ('D' for daily, 'W' for weekly, 'M' for monthly) or 'Y' for yearly.
        log_returns (bool, optional): Whether to calculate logarithmic returns (True) or simple returns (False). Defaults to False.

    Raises:
        ValueError: Select correct frequency

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the calculated returns.
    """
    
    if log_returns:
        returns = np.log(data).diff(1)
        method = aggregate_logarithmic_returns
    else:
        returns = data.pct_change()
        method = aggregate_simple_returns

    if drop_na:
        returns = returns.dropna()

    if frequency == 'D':
        returns = returns
    elif frequency == 'W':
        returns = returns.resample('W').apply(method)
    elif frequency == 'M':
        returns = returns.resample('M').apply(method)
    elif frequency == 'Y':
        returns = returns.resample('Y').apply(method)
    else:
        raise ValueError("Invalid frequency specified. Please choose 'D', 'W', 'M' or 'Y'.")
    
    if drop_zeros:
        returns = returns.loc[~(returns == 0).all(axis=1)]      # usually first entry while resampling is zero due to lack of data

    if demean:
        returns = (returns - returns.mean())/returns.std()

    return returns

