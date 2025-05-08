import pandas as pd
import numpy as np
import functions.matrix_algebra as malg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.stats import norm
from typing import Optional

from functions.utilities import spectral_correction, correlation_bounds_correction, get_correlation_pair_names
from functions.fin_econ import filter_garch_effects, introduce_garch_effects, compute_tail_risk
from functions.sampling import create_bootstrap_sample, sample_asset_portfolio_quantiles


pd.options.display.float_format = '{:.4f}'.format
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})


class QuantileImpliedCorrelation:
    def __init__(self, 
                 returns: pd.DataFrame, 
                 benchmark_portfolios_weights,
                 ):
        
        self.returns = returns
        self.tickers = returns.columns
        self.n_assets = self.returns.shape[1]
        self.benchmark_portfolios = benchmark_portfolios_weights
        self.n_portfolios = len(self.benchmark_portfolios)
        self.asset_alpha = None
        self.asset_risk = None
        self.portfolio_alpha = None
        self.portfolio_risk = None
        self.tail_area = None
        self.QIC = None
        self._QIC_vecl = None
        self.QIC_SE = None
        self.gradient = None
        self.QIC_CI = None
        self.truncation = None
        self.spectral_correction = None
        

        
    @staticmethod
    def _compute_QIC(asset_returns: pd.DataFrame, 
                     asset_alphas: list, 
                     asset_risk_metric: str,
                     portfolios_weights: np.array, 
                     portfolio_alphas: list, 
                     portfolio_risk_metric: str, 
                     tail: str,
                     truncate=True, 
                     spectral_correct=True, 
                     derivative_output=False):
        """
        Compute quantile or expectile implied correlation matrix.

        Args:
            asset_returns (pd.DataFrame): financial return data
            asset_alphas (list): alpha-quantile levels for each assets (needs to be implemented for assets seperatly)
            asset_risk_metric (str): risk metric to determine the quantile/expectile, i.e. VaR, ES
            portfolios_weights (np.array): benchmark portfolios weights
            portfolio_alphas (list): alpha-quantile levels for each portfolio (needs to implement for portfolios seperatly)
            portfolio_risk_metric (str): risk metric to determine the quantile/expectile, i.e. VaR, ES
            tail (str): specify tail area, 'upper' or 'lower'
            truncate (bool, optional): truncate correlation coef. to 1 or -1 if exceeds boundary. Defaults to True.
            spectral_correct (bool, optional): Make matrix positive semi-definite. Defaults to True.
            output_deri (bool, optional): Return components for gradient computation. Defaults to False.

        Returns:
            ndarray: implied correlation matrix
        """
        portfolio_returns = asset_returns.dot(portfolios_weights.T)

        # not robust for small sample sizes, need to implement correction
        asset_quantiles = compute_tail_risk(asset_returns, asset_alphas, asset_risk_metric, tail)
        portfolio_quantiles = compute_tail_risk(portfolio_returns, portfolio_alphas, portfolio_risk_metric, tail)

        n_assets = len(asset_quantiles)

        q_tilde_p = np.square(portfolio_quantiles).T - np.square(portfolios_weights).dot(np.square(asset_quantiles).T)
        Z_alpha = asset_quantiles * portfolios_weights
        D = malg.duplicate_vecl(n_assets)
        X = malg.generalized_kronecker(Z_alpha, Z_alpha, 'row').dot(D)

        quantile_implied_correlation_vector = np.linalg.lstsq(X, q_tilde_p, rcond=-1)[0]
        qimp_matrix = malg.antivecl(quantile_implied_correlation_vector.flatten())

        if truncate:
            if qimp_matrix.max() > 1 or qimp_matrix.min() < -1:
                qimp_matrix = correlation_bounds_correction(qimp_matrix)

        if spectral_correct:
            if not malg.is_positive_semi_definite_matrix(qimp_matrix):
                qimp_matrix = spectral_correction(qimp_matrix)

        if derivative_output:
            return malg.vecl(qimp_matrix), X, D, Z_alpha, asset_quantiles, portfolio_quantiles
        else:
            return qimp_matrix
        

    def _fill_matrix_with_QIC_and_SE(self, QIC_estimate, QIC_SE_estimate):
        """
        Generate correlation matrix with SE in brackets.
            - create dataframe with zeros according to number of assets
            - fill matrix with QIC estimates and QIC SE estimates

        Args:
            QIC_estimate (np.array): QIC estimate in vecl form
            QIC_SE_estimate (np.array): QIC SE estimates, from theoretical, garch and simulation methods

        Returns:
            pd.DataFrame: Pandas dataframe with QIC and SE in brackets.
        """

        df = pd.DataFrame(np.zeros((self.n_assets, self.n_assets)) + np.identity(self.n_assets), 
                                    columns = self.tickers, 
                                    index= self.tickers)

        entries = [str(round(qic, 3)) + ' ' + '(' + str(round(qic_se, 3))+ ')' for qic, qic_se in zip(QIC_estimate, QIC_SE_estimate)]

        ro, co = np.triu_indices(self.n_assets, 1)

        i = 0
        for r,c in zip(ro,co):
            df.iloc[r,c] = entries[i]
            i += 1
        
        df.replace(0.0, '', inplace=True)
        
        return df.T
        
    
    def _compute_asset_portfolio_covariance(self, iteration_size):
        """
        Generate covariance matrix of asset quantiles and portfolio quantiles.
        Covariance matrix used with gradient later for theoretical standard errors

        Args:
            iteration_size (int): bootstrap sample size

        Returns:
            pd.Dataframe: Covariance matrix of asset and portfolio quantiles
        """
        
        if iteration_size is None:
            iteration_size = 300

        a_p_sample = sample_asset_portfolio_quantiles(self.returns, 
                                                      self.benchmark_portfolios, 
                                                      self.asset_alpha, iteration_size = iteration_size)
        
        a_p_sample = pd.DataFrame(a_p_sample)
        
        return a_p_sample.cov()
    
    
    def _compute_QIC_gradient(self):
        """
        See personal notes for derivation of the gradient. Sparse matrices are needed w.r.t. to the permutation and kommutation matrices
        
        See: https://medium.com/codex/a-quick-guide-to-operations-on-sparse-matrices-2f8776fab265

        Returns:
            dict: dictionary with different alpha levels as keys, each containing the gradient w.r.t. to selected alpha level
        """
        
        asset_tail_lvl = self.n_assets * [self.asset_alpha]
        portfolios_tail_lvl = self.n_portfolios * [self.portfolio_alpha]

        rho, X, D, Z_alpha, asset_quantiles, portfolio_quantiles = self._compute_QIC(self.returns.values, 
                                                                                    asset_tail_lvl, 
                                                                                    self.asset_risk,
                                                                                    self.benchmark_portfolios, 
                                                                                    portfolios_tail_lvl, 
                                                                                    self.portfolio_risk,
                                                                                    self.tail_area,
                                                                                    derivative_output=True)
        
        
        X_inv = sparse.csc_matrix(np.linalg.inv(X))
        D = sparse.csc_matrix(D)
        S = malg.khatri_kronecker_selection_matrix(self.n_portfolios, make_sparse=True).transpose()
        Z = sparse.csc_matrix(Z_alpha)


        # Derive eq. (29)
        d_asset_q = -2 * np.tile(asset_quantiles,(self.n_portfolios,1)) * self.benchmark_portfolios**2
        d_portfolio_q = np.diag(portfolio_quantiles)
        D_qp = np.concatenate((d_asset_q, d_portfolio_q), axis=1)


        ## Derivate D_A_alpha in eq. (34)
        # Generate list of partial derivatives w.r.t. to each asset quantile in sparse matrix format
        def generate_dW_matrix(X, w, col):
            X[:,col] = w
            return X

        init_dW = np.zeros((self.n_portfolios, self.n_assets + self.n_portfolios))     # we need asset and portfolio quantiles
        lst_dW = [generate_dW_matrix(init_dW.copy(), self.benchmark_portfolios[:,col], col) for col in range(self.n_assets)]
        lst_dW = list(map(sparse.csc_matrix, lst_dW))


        # Compute derivative with chain rule
        lst_D_A_alpha = []
        for outer_col in range(self.n_assets):
            for inner_col in range(self.n_assets):
                dZ_kron_Z = sparse.kron(lst_dW[outer_col], Z[:, inner_col])
                Z_kron_dZ = sparse.kron(Z[:, outer_col], lst_dW[inner_col])
                lst_D_A_alpha.append(dZ_kron_Z + Z_kron_dZ)

        D_A_alpha = sparse.vstack(lst_D_A_alpha)  


        # Coefficient of D_A_alpha, eq (34)
        coef_A = sparse.kron(rho,X_inv) @ sparse.kron(D.transpose(), S)
        part_2 = coef_A @ D_A_alpha
        part_2 = np.array(part_2.todense())

        self.gradient = (X_inv @ D_qp) - part_2
        
        return self.gradient
    
        
    def _sample_QIC_with_garch_filtered_returns(self, iteration_size):
        """
        - filter heteroscedastocoty in returns with GARCH(1,1) normal model, yields iid returns
        - create bootstrap sample iid returns
        - reintroduce heteroscedastocoty in returns
        - compute QIC based on new sample
        - store vectorized QIC matrix in a list
 
        Args:
            iteration_size (int, optional): Number of iterations. Defaults to 200.

        Raises:
            NameError: Base case must run first, so the parameters can be passed to the QIC function 
            inside the iteration.
        """
        
        if iteration_size is None:
            iteration_size = 300
        
        garch_filtered_returns, garch_models_params = filter_garch_effects(self.returns)                       # hard coded normal constant garch(1,1)
        garch_params_dict = {key:garch_models_params[key].params for key in garch_models_params.keys()} 
        
        res = []
        
        print(f'Number of bootstrap Iteration: {iteration_size}')
        
        burn_in = 500
    
        for i in range(iteration_size):
            # IID sample
            boot_sample_iid = create_bootstrap_sample(garch_filtered_returns.values, self.returns.shape[0]+burn_in)
            boot_sample_iid = dict(zip(self.tickers, boot_sample_iid.T))

            # Sample with GARCH11 effects
            boot_sample_garch = {key: introduce_garch_effects(garch_params_dict[key], boot_sample_iid[key], burn_in=burn_in) for key in garch_params_dict.keys()}
            boot_sample_garch = pd.DataFrame(boot_sample_garch)

            QIC_garch = QuantileImpliedCorrelation(boot_sample_garch, self.benchmark_portfolios)
            
            res.append(malg.vecl(QIC_garch.qic(self.asset_alpha, 
                                               self.asset_risk, 
                                               self.portfolio_risk, 
                                               self.tail_area).values))
            
        return np.array(res)
    
    
    def _sample_QIC_with_raw_returns(self, iteration_size=None):
        """
        - create bootstrap sample raw returns
        - compute QIC based on new sample
        - store vectorized QIC matrix in a list
 
        Args:
            iteration_size (int, optional): Number of iterations. Defaults to 200.

        """
        
        if iteration_size is None:
            iteration_size = 300
        
        res = []
        
        print(f'Number of bootstrap Iteration: {iteration_size}')
    
        for i in range(iteration_size):
            boot_sample_raw = create_bootstrap_sample(self.returns.values, self.returns.shape[0])
            boot_sample_raw = dict(zip(self.tickers, boot_sample_raw.T))
            boot_sample_raw = pd.DataFrame(boot_sample_raw)

            QIC_raw = QuantileImpliedCorrelation(boot_sample_raw, self.benchmark_portfolios)
            
            res.append(malg.vecl(QIC_raw.qic(self.asset_alpha, 
                                               self.asset_risk, 
                                               self.portfolio_risk, 
                                               self.tail_area).values))
            
        return np.array(res)    
    
    
    def _sample_QIC_with_simulated_returns(self, iteration_size):
        """
        - estimate covariance and mean of return sample
        - simulate returns based parameters above, normal distribution selected

        Returns:
            mean and standard error of simulated sample of all iterations
        """
        
        if iteration_size is None:
            iteration_size = 300
        
        
        # empirical covariance and mean
        cov_mat = self.returns.cov()
        mean_vec = self.returns.mean()
        sample_size = self.returns.shape[0]

        res = []
        
        print(f'Number of simulated returns iteration: {iteration_size}')
        
        for i in range(iteration_size):

            simulated_returns = pd.DataFrame(np.random.multivariate_normal(mean_vec, cov_mat, sample_size),
                                             columns=self.tickers)

            QIC_normal_sim = QuantileImpliedCorrelation(simulated_returns, self.benchmark_portfolios)

            res.append(malg.vecl(QIC_normal_sim.qic(self.asset_alpha, 
                                                    self.asset_risk, 
                                                    self.portfolio_risk, 
                                                    self.tail_area).values))
            
        res = np.array(res)        

        return np.mean(res, axis = 0), np.std(res, axis = 0)   


    def _CI_percentile(self, alpha, iteration):
        sample_garch_QIC = self._sample_QIC_with_garch_filtered_returns(iteration)
        lower_crit_val = np.quantile(sample_garch_QIC, alpha*0.5, axis=0, method='closest_observation')
        upper_crit_val = np.quantile(sample_garch_QIC, 1-(alpha*0.5), axis=0, method='closest_observation')
        return lower_crit_val, upper_crit_val
           
            
    def _CI_pivotal(self, alpha, iteration):
        sample_garch_QIC = self._sample_QIC_with_garch_filtered_returns(iteration)
        QIC_dispersion = malg.vecl(self.QIC.values) - sample_garch_QIC
        lower_crit_val = np.quantile(QIC_dispersion, 1-(alpha*0.5), axis=0, method='closest_observation')
        upper_crit_val = np.quantile(QIC_dispersion, alpha*0.5, axis=0, method='closest_observation')
        return self._QIC_vecl - lower_crit_val, self._QIC_vecl - upper_crit_val               
    
    
    def _CI_theoretical(self, alpha, iteration):
        """See https://www.statskingdom.com/correlation-confidence-interval-calculator.html
        
           Otherwise the confidence intervals exceed 1

        Args:
            alpha (_type_): _description_
            iteration (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.QIC_SE is None:
            self._compute_QIC_gradient()
            quantile_covariance = self._compute_asset_portfolio_covariance(iteration)
            theoretical_covmat = self.gradient @ quantile_covariance @ self.gradient.T
            self.QIC_SE = np.sqrt(np.diag(theoretical_covmat))
        
        dist = norm(loc=0, scale=1)
        z = dist.ppf(alpha)                              # double check z-score on confidence intervals
            
        fisher_transformed_QIC = np.arctanh(self._QIC_vecl)
        f_lower_bound = fisher_transformed_QIC - (z * self.QIC_SE)
        f_upper_bound = fisher_transformed_QIC + (z * self.QIC_SE)
                          
        lower_bound = np.tanh(f_lower_bound)
        upper_bound = np.tanh(f_upper_bound)
        
        return lower_bound, upper_bound


    def qic(self, alpha: float, 
            asset_risk: str, portfolio_risk: str, tail_area: str,
            SE: Optional[str] = None, 
            iteration: Optional[int] = None) -> pd.DataFrame:
        """
        Compute the VaR or ES implied correlation for a specified tail given some alpha levels.
        Depending on the choice of SE, the SE are either derived theoretically (i.e. by asymptotics) or via garch filtered bootstrap or simulation.

        Args:
            alpha (float): Quantile level for all assets and portfolios, alpha in (0,1)
            asset_risk (str): Risk measure for assets, VaR or ES
            portfolio_risk (str): Risk measure for all portfolios, VaR or ES
            tail_area (str): Specify tail area, lower or upper
            SE (Optional[str], optional): Derive Standard Errors theoretically, garch filter or by simulation. Defaults to None.
            iteration (Optional[int], optional): Only if SE is specified, set the amount of bootstrap iteration. For the theoretical error, iteration specifies the bootstrap iteration for estimating the asset and portfolio quantile covariance matrix. For garch filter and simulation, iteration specifies the amount of bootstrap QICs computed.  Defaults to None.

        Returns:
            pd.DataFrame: Implied Correlation matrix as a dataframe with or without Standard Errors
        """

        
        assert SE in [None, 'garch', 'theoretical', 'simulated'], 'Wrong SE type specified'
                
        asset_tail_lvl = self.n_assets * [alpha]
        portfolios_tail_lvl = self.n_portfolios * [alpha]

        QIC = self._compute_QIC(self.returns.values, 
                                asset_tail_lvl, 
                                asset_risk,
                                self.benchmark_portfolios, 
                                portfolios_tail_lvl, 
                                portfolio_risk,
                                tail_area)
        
        QIC_matrix = pd.DataFrame(data=QIC, columns = self.tickers, index = self.tickers)
    
    
        self.QIC = QIC_matrix
        self._QIC_vecl = malg.vecl(self.QIC.values)
        self.asset_alpha = alpha
        self.asset_risk = asset_risk
        self.portfolio_alpha = alpha                            # more granularity to come
        self.portfolio_risk = portfolio_risk
        self.tail_area = tail_area
 
        
        if SE == None:
            return QIC_matrix
        elif SE == 'garch':     
            bootstrap_sample_garch_QIC = self._sample_QIC_with_garch_filtered_returns(iteration)
            self.QIC_SE = np.std(bootstrap_sample_garch_QIC, axis = 0)
        elif SE == 'theoretical':
            self._compute_QIC_gradient()
            quantile_covariance = self._compute_asset_portfolio_covariance(iteration)
            theoretical_covmat = self.gradient @ quantile_covariance @ self.gradient.T
            self.QIC_SE = np.sqrt(np.diag(theoretical_covmat))
        elif SE == 'simulated':
            _ , self.QIC_SE = self._sample_QIC_with_simulated_returns(iteration)
            
        QIC_SE_mat = self._fill_matrix_with_QIC_and_SE(self._QIC_vecl, self.QIC_SE)
        
        return QIC_SE_mat
    
    
    def qic_confidence_interval(self, conf_lvl, iteration=None, method='pivotal'):
        """
        See methods
            https://stats.stackexchange.com/questions/355781/is-it-true-that-the-percentile-bootstrap-should-never-be-used
            https://faculty.washington.edu/yenchic/17Sp_403/Lec5-bootstrap.pdf
            
            and Larry Wassermann all of statistics

        Args:
            conf_lvl (_type_): _description_
            iteration (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        alpha = 1-conf_lvl
        
        assert self.QIC is not None, 'Run base QIC first'
        assert method in ['pivotal', 'percentile', 'theoretical'], 'Wrong confidence interval method specified'

        # get name of correlation pairs for labeling
        corr_pair_mat = get_correlation_pair_names(self.tickers)
        corr_pair_names = corr_pair_mat.T[np.triu_indices(corr_pair_mat.shape[0], 1)]
        
        data_dict = {}
        data_dict['asset pairs'] = corr_pair_names        
        
        # See Wasserman(2004, 'All of Statistics', p.111)
        if method == 'pivotal':
            data_dict['lower'], data_dict['upper'] = self._CI_pivotal(alpha, iteration)
        elif method == 'percentile':
            data_dict['lower'], data_dict['upper'] = self._CI_percentile(alpha, iteration)
        elif method == 'theoretical':
            data_dict['lower'], data_dict['upper'] = self._CI_theoretical(alpha, iteration)
        
        self.QIC_CI = pd.DataFrame(data_dict)
        self.QIC_CI['qic'] = self._QIC_vecl
        self.QIC_CI['in_range'] = np.where((self.QIC_CI['qic'] >= self.QIC_CI['lower']) & (self.QIC_CI['qic'] <= self.QIC_CI['upper']), True, False)
        
        return self.QIC_CI        
        
    
    def plot_QIC_heatmap(self, title=False, annot=False, save_plot=False, save_path = None):
        
        assert self.QIC is not None, 'Run base QIC first'
        
        fig, ax = plt.subplots()
        if title:
            ax.set_title(f'{self.asset_alpha} - {self.asset_risk} Implied Correlation - {self.tail_area} tail',
                        pad = 10)
        fig.set_size_inches(10, 7)
        sns.heatmap(self.QIC, 
                    annot=annot,
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1, 
                    xticklabels=True, 
                    yticklabels=True,
                    ax=ax)
        # ax.set_aspect('equal', adjustable='box')        # make plot square
        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path, dpi=500)
            
        plt.show()
        
        return fig, ax
        

    def plot_QIC_confidence_interval(self, save_plot=False):
        
        assert self.QIC_CI is not None, 'Run confidence interval method first'
        
        pearson_vecl = malg.vecl(self.returns.corr().values)
        pearson_mean = np.mean(pearson_vecl)
        
        fig, ax = plt.subplots(figsize=(8,self.n_assets*1.6))
        
        for lower,upper,y in zip(self.QIC_CI['lower'],self.QIC_CI['upper'],range(len(self.QIC_CI))):
            plt.plot((lower,upper),(y,y),'|-',color='orange')
        for qic, y in zip(self._QIC_vecl, range(len(self.QIC_CI))):
            plt.plot(qic, y, 'p', color='red')
        for pear, y in zip(pearson_vecl, range(len(self.QIC_CI))):
            plt.plot(pear, y, '1', color='blue')
        
        plt.yticks(range(len(self.QIC_CI)),list(self.QIC_CI['asset pairs']))
        plt.axvline(pearson_mean, ls='--')
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        
        if save_plot:
            fig_name = f'{self.asset_risk}_{str(self.asset_alpha).replace(".", "")}_{self.tail_area}_conf_interval.pdf'
            plt.savefig(fname=fig_name, bbox_inches='tight', dpi=500)
        
        plt.show()
        
        return fig, ax
        
        
    @classmethod  
    def QIC_heatmap(clc, QIC, asset_alpha, asset_risk, tail_area, title=False, annot=False, save_plot=False, save_path = None):
        """
        Used for results processing
        """
        
        fig, ax = plt.subplots()
        if title:
            ax.set_title(f'{asset_alpha}-{asset_risk} Implied Correlation - {tail_area} tail',
                        pad = 10)
        fig.set_size_inches(10, 7)
        sns.heatmap(QIC, 
                    annot=annot,
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1, 
                    xticklabels=True, 
                    yticklabels=True,
                    ax=ax)
        # ax.set_aspect('equal', adjustable='box')        # make plot square
        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path, dpi=500)
            
        plt.show()
        
        return fig, ax
    
    
    @classmethod
    def matrix_with_QIC_and_SE(clc, QIC_estimate, QIC_SE_estimate, tickers):

        df = pd.DataFrame(np.zeros((len(tickers), len(tickers))) + np.identity(len(tickers)), 
                                    columns = tickers, 
                                    index= tickers)

        entries = [str(round(qic, 3)) + ' ' + '(' + str(round(qic_se, 3))+ ')' for qic, qic_se in zip(QIC_estimate, QIC_SE_estimate)]

        ro, co = np.triu_indices(len(tickers), 1)

        i = 0
        for r,c in zip(ro,co):
            df.iloc[r,c] = entries[i]
            i += 1
        
        df.replace(0.0, '', inplace=True)
    
        return df.T
    
    
    @classmethod
    def matrix_with_QIC_and_SE_latex(clc, QIC_estimate, QIC_SE_estimate, tickers):

        df = pd.DataFrame(np.zeros((len(tickers), len(tickers))) + np.identity(len(tickers)), 
                                    columns = tickers, 
                                    index= tickers)

        entries = [r'\makecell{' + str(round(qic, 3)) + r' \\ (' + str(round(qic_se, 3)) + r')}' for qic, qic_se in zip(QIC_estimate, QIC_SE_estimate)]

        ro, co = np.triu_indices(len(tickers), 1)

        i = 0
        for r,c in zip(ro,co):
            df.iloc[r,c] = entries[i]
            i += 1
        
        df.replace(0.0, '', inplace=True)
    
        return df.T
    
    
def qic_with_quantiles(asset_quantiles, 
                       portfolio_quantiles, 
                       portfolios_weights: np.array, 
                       truncate=True, 
                       spectral_correct=True, 
                       derivative_output=False):
        """
        Used for correlation forecasting, where the quantiles are separately entered from CAVIAR models

        Args:
            asset_quantiles (_type_): _description_
            portfolio_quantiles (_type_): _description_
            portfolios_weights (np.array): _description_
            truncate (bool, optional): _description_. Defaults to True.
            spectral_correct (bool, optional): _description_. Defaults to True.
            derivative_output (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """


        n_assets = len(asset_quantiles)

        q_tilde_p = np.square(portfolio_quantiles).T - np.square(portfolios_weights).dot(np.square(asset_quantiles).T)
        Z_alpha = asset_quantiles * portfolios_weights
        D = malg.duplicate_vecl(n_assets)
        X = malg.generalized_kronecker(Z_alpha, Z_alpha, 'row').dot(D)

        quantile_implied_correlation_vector = np.linalg.lstsq(X, q_tilde_p, rcond=-1)[0]
        qimp_matrix = malg.antivecl(quantile_implied_correlation_vector.flatten())

        if truncate:
            if qimp_matrix.max() > 1 or qimp_matrix.min() < -1:
                qimp_matrix = correlation_bounds_correction(qimp_matrix)

        if spectral_correct:
            if not malg.is_positive_semi_definite_matrix(qimp_matrix):
                qimp_matrix = spectral_correction(qimp_matrix)

        if derivative_output:
            return malg.vecl(qimp_matrix), X, D, Z_alpha, asset_quantiles, portfolio_quantiles
        else:
            return qimp_matrix