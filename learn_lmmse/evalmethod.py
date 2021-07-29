import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from itertools import cycle


def lstsq_estimation(Y_train, X_train):
    '''Computes the least squares estimator for n input-output pairs (Y,X).
    
    Parameters:
        Y_train: ndarray of shape (n, dim_Y)
            matrix containing n realizations of the input data Y
        X_train: ndarray of shape (n,dim_X)
            matrix containing n realizations of the output prior X
    
    Returns:
        hat_LMMSE : function()
            The linear least squares estimator. Takes ndarray of shape
            (k,dim_Y) of k realization of Y as inputs and returns 
            the corresponding estimates for X in the form of an (k,dim_X) 
            ndarray
    '''
    hat_theta = linalg.lstsq(Y_train, X_train)[0]    
    def hat_LMMSE(Y):
        '''Least squares estimator of X given Y. 
        
        Parameters:
            Y : ndarray of shape (k, dim_Y)
        
        Returns:
            X : ndarray of shape (k, dim_Y)
        '''
        return Y @ hat_theta
    return hat_LMMSE


def eval_LMMSE_approx_method(
        my_inverse_problem, 
        methods=None,
        n_exp=100, 
        n_train=20, 
        n_test=2000,  
        test_seed=0, 
        train_seed_init=1
        ):
    '''Computes empirical tail distributions of errors for estimation methods.
    
    For a given inverse problem and user-specified linear estimation 
    method, this function computes the empirical tail distributions of the 
    following errors:
        the componentwise train error,
        the componentwise test error,
        the componentwise mean squared error.
    
    Parameters:
        my_inverse_problem : class 
            instance of a subclass of InverseProblem
        methods : dictionary
            Dictionary of methods for obtaining linear estimators which should 
            be tested. If no methods are provided, least squares estimation is 
            used. Each method in methods.values() should be a function of the 
            form:
            def method(Y,X)
                    Parameters:
                        Y_train: ndarray of shape (n, dim_Y)
                            n realizations of the input Y
                        X_train: ndarray of shape (n,dim_X)
                            n realizations of the output prior X
                    Returns:
                        hat_LMMSE : function()
                        Estimator with ndarray of shape (k,dim_Y) as parameter
                        and (k,dim_X) ndarray as return value. 
        n_exp : int
            number of experiments to obtain empirical tail distribution
        n_train : int
            number of training samples used in each experiment
        n_test : int
            number of test_samples used in each experiment
        test_seed : int
            seed for the test set
        train_seed_init : int
            The seed which is used for the 0th training set. The ith training 
            set is using the seed train_seed_init+i.
    
    Returns:
        results : dictionary
            Returns the componentwise errors for provided methods.
            For name in methods.keys(), we have
                results['train_error_' + name]: ndarray of shape (n_exp,dim_X)
                    Componentwise training error.
                results['test_error_' + name]: ndarray of shape (n_exp,dim_X)
                    Componentwise test error.
                results['mse_' + name]: ndarray of shape (n_exp,dim_X)
                    Componentwise mean squared error.
    '''
    if methods is None:
        # the default method is least squares estimation
        methods = {}
        methods['lstsq'] = lstsq_estimation
        
    dim_X = my_inverse_problem.dim_X
    results = {}
    
    for name in methods.keys():
        results['train_error_' + name] = np.zeros((n_exp,dim_X))
        results['test_error_' + name] = np.zeros((n_exp,dim_X))
        results['mse_' + name] = np.zeros((n_exp,dim_X))
        
    L = np.linalg.cholesky(my_inverse_problem.cov_Y)
    X_test, Y_test, Z_test = my_inverse_problem.create_samples(
        n_test, 
        seed=test_seed,
        sample_type='test'
        )
    for it_exp in range(n_exp):
        if train_seed_init is not None:
            train_seed = train_seed_init + it_exp
        else:
            train_seed = None
       
        X_train, Y_train = my_inverse_problem.create_samples(
                               n_train, 
                               seed=train_seed
                                )[:2]
        for name, method in methods.items():
            hat_LMMSE = method(Y_train,X_train)            
            results['train_error_' + name][it_exp,:] = (
                np.mean((hat_LMMSE(Y_train)-X_train)**2, axis=0)
                )
            results['test_error_' + name][it_exp,:] = (
                np.mean((hat_LMMSE(Y_test)-X_test)**2, axis=0)
                )
            results['mse_' + name][it_exp,:] = (
                np.sum((hat_LMMSE(L.T)-L.T@my_inverse_problem.LMMSE)**2, 
                       axis=0) 
                + np.diag(my_inverse_problem.cov_E)
                )
    return results


def eval_gaussian_prediction(
        my_inverse_problem, 
        epsilons=None,
        n_exp=300, 
        n_test=10000,
        test_seed=None, 
        train_seed_init=None
        ):
    '''Plots tail distribution and predicted expected value of squared error.

    For epsilon \in epsilons, the number of samples n is chosen according to 
    n = dim_Y/epsilon + dim_Y + 1. The details for the experiment can be found
    in Section 6.1 of G. Holler, "How many samples are needed to reliably 
    approximate the best linear estimator for a linear inverse problem?",
    arXiv preprint arXiv:2107.00215, 2021, which is available at
    https://arxiv.org/abs/2107.00215.
    
    Parameters:
        my_inverse_problem : class 
            instance of subclass of InverseProblem
        epsilons : ndarray of shape (k,)
            targets for relative tolerances
        n_exp : 
            number of experiments to obtain empirical tail distribution
        n_test : int
            number of test_samples used in each experiment
        test_seed : int
            seed for the test set
        train_seed_init : int
            The seed which is used for the 0th training set. The ith training 
            set is using the seed train_seed_init+i.

    Returns:
        my_data: dictionary
            Dictionary which contains the results of the experiment. The errors
            can be accessed via my_data['results'].
    '''
    if epsilons is None:
        epsilons = np.array([1/16, 1/4, 1/2, 1])

    dim_Y = my_inverse_problem.dim_Y
    n_train_vec = np.ceil(dim_Y/epsilons).astype('int') + dim_Y + 1
    
    my_data = {}
    my_data['inverse_problem'] = type(my_inverse_problem).__name__
    my_data['epsilons'] = epsilons
    my_data['dim_Y'] = dim_Y
    my_data['dim_X'] = my_inverse_problem.dim_X
    my_data['MSE_LMMSE'] = my_inverse_problem.MSE
    results_dic = {}
    for n_train in n_train_vec:
        results_dic[str(n_train)] = eval_LMMSE_approx_method(
                                        my_inverse_problem, 
                                        n_exp=n_exp, 
                                        n_train=n_train, 
                                        n_test=n_test,
                                        test_seed=None, 
                                        train_seed_init=None
                                        )
    my_data['results'] = results_dic
    return my_data


def plot_tail_basic(
        ax,
        my_data,
        method_name='lstsq',
        error_type='mse_',
        coord='sum',    
        **kwargs
        ):
    '''Plots the empirical tail distribution for a given error of a method.   
    
    Parameters:
        ax : 
            axis where the tail distribution should be plotted
        my_data : dictionary
            dictionary containing the results
        method_name :
            method whose error is plotted.
        error_type : string
            error_type whose empirical tail distribution should be plotted  
        coord : int or string
            coordinate of error which should be plotted 
            (default is 'sum' over all coordinates)
        **kwargs :    
            Arbitrary keyword arguments which are passed on to 
            matplotlib.pyplot.plot function
    '''
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for n_train, results in my_data['results'].items():
            if coord is 'sum':
                data = np.sum(results[error_type+method_name], axis=1)
            else:
                data = results[error_type+method_name][:,coord]
            
            ax.plot(
                np.sort(data), 
                np.linspace(1,0,len(data), 
                            endpoint=False),
                next(linecycler),
                label='$n=$'+n_train,
                **kwargs,
                )       

            
def plot_tail_fancy(
        ax,
        my_data,
        method_name='lstsq',
        error_type='mse_', 
        fontsize=24,
        tick_labelsize=24,
        labelsize=34,
        coord='sum',
        **kwargs
        ):
    '''Plots the tail distribution for the mean squared error or test error.
    
    Parameters:
        ax : 
            axis where the tail distribution should be plotted
        my_data : dictionary
            dictionary containing the results
        method_name :
            method whose error is plotted.
        error_type : string
            error_type whose empirical tail distribution should be plotted  
        fontsize : 
            fontsize for the legend
        tick_labelsize :
            fontsize for xticks and yticks
        labelsize :
            fontsize for xlabels and ylabels
        coord :
            coordinate of error which should be plotted 
            (default is 'sum' over all coordinates)
        **kwargs :    
            Arbitrary keyword arguments which are passed on to the
            matplotlib.pyplot.plot function
    '''
    plot_tail_basic(ax=ax,
                    my_data=my_data,
                    method_name=method_name,
                    error_type=error_type,
                    coord=coord,
                    **kwargs)

    epsilons = my_data['epsilons']
    expected_mses =  (1+epsilons) * my_data['MSE_LMMSE']

    for xc in expected_mses:
        ax.axvline(x=xc, linestyle='--', c='gray')
    labels =[]
    for it in range(len(epsilons)):
        labels.append(str(1+epsilons[it]))

    ax.set_xticks(expected_mses) 
    ax.set_xticklabels(labels)
    ax.set_yticks([0.1,0.3,0.5,0.7,0.9])
    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)
    ax.set_xlabel('$\\tau$', fontsize=labelsize)
    if error_type == 'test_error_':
        ax.set_ylabel('$\mathrm{\mathbb{P}}(\mathrm{MSE}_{\mathrm{test}}' 
                      + '(\hat{\mathsf{f}})>\\tau \cdot ' 
                      + '\mathrm{tr}(C_{EE}))$', 
                      fontsize=labelsize)
    elif error_type == 'mse_':
        ax.set_ylabel('$\mathrm{\mathbb{P}}(\mathrm{MSE}(\hat{\mathsf{f}})'
                      +'>\\tau \cdot \mathrm{tr}(C_{EE}))$', 
                      fontsize=labelsize)        

    ax.set_ylim([0,1])
    ax.set_xlim([0.95*expected_mses[0], 
                 expected_mses[-1]+(1/2)*expected_mses[0]])
    ax.legend(fontsize=fontsize)