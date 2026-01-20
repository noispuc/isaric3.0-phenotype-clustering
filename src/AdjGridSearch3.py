""" Adjusted grid search function for stepmix 
[https://github.com/Labo-Lacourse/stepmix/blob/master/stepmix] 
Major improvement: display several metrics for consideration,
alongside the convergence status, with visual plots to enable 
better decision-making
Not yet exactly a "grid" search, as just one step condition will be explored

v3: add log-likelihood test for nested models
v2: store object, report convergence
v1: other metrics beyond log-likelihood is reported
for now, not yet implemented:
- log-likelihood tests;
- for different nsteps (just the default n_steps=1)

"""

import itertools
import pandas as pd
import warnings
import copy
from scipy.stats import norm
import matplotlib.pyplot as plt

import numpy as np
import tqdm

from sklearn.base import clone
from sklearn.utils.validation import check_random_state, check_is_fitted

from scipy.stats import chi2

def lrt(null_ll, alternative_ll, null_dof, alternative_dof):
    """LRT Test    

    Parameters
    ----------
    null_ll: log-likelihood with k classes.
    alternative_ll : log-likelihood with k + 1 classes.   

    Returns
    ----------
    p-value: float
        p-value of the LRT test. A significant test indicates the alternative k + 1 model provides a
        significantly better fit of the data.
    """
    
    LR = 2 * (alternative_ll - null_ll)     
    dof_diff=null_dof-alternative_dof
    
    p_value = chi2.sf(LR, dof_diff)
    return dict({'log-likelihood diff': LR, 'dof diff': dof_diff, 'p value': p_value})

def gridSearch(
    model, X, Y=None, low=1, high=5, random_state=42, verbose=True, sample_weight=None
):
    """GridSearch for LCA optimization

    Get fit measures for a range of number of classes. For example, if you set low=1 and high=4, the function
    will return the metrics log-likelihood, AIC, BIC for each of the number of classes from 1 to 4. Note those are point-estimates, and no bootstrapping is applied.

    Parameters
    ----------
    model : StepMix instance
        A StepMix model.
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None
    low: int, default=1
        Minimum number of classes to test.
    high: int, default=5
        Maximum number of classes to test.    
    random_state : int, default=None
    verbose : bool, default=True

    Returns
    ----------
    p-values: DataFrame
        p-values of the BLRT test for each comparison.
    """
    test_string = list()
    p_values = list()
    stats=list()
    obj_list=list()
    n_samples = X.shape[0]
    
    for k in range(low, high):
        print(f"Testing {k} classes...")
        estimator_rep = clone(model)
        estimator_rep.set_params(n_components=k)
        estimator_rep.set_params(verbose=0) # added to check if it stops printing report
        estimator_rep.fit(X, Y, sample_weight=sample_weight)
        avg_ll = estimator_rep.score(X, Y, sample_weight=sample_weight)
        
        ll = (
            avg_ll * np.sum(sample_weight)
            if sample_weight is not None
            else avg_ll * n_samples
        )
        npar=estimator_rep.n_parameters
        n = X.shape[0]
        ncomp=estimator_rep.n_components
        aic=( -2 * avg_ll * n + 2 * npar)
        bic=(-2 * avg_ll * n + npar * np.log( n ))        
        caic=( -2 * avg_ll * n + npar * (np.log(n) + 1))
        sabic=(-2 *avg_ll * n + npar * np.log(
            n * ((n + 2) / 24)
        ))
        entropy=estimator_rep.entropy(X)
        relentropy= (
            1 - entropy / (n * np.log(ncomp))
            if ncomp > 1
            else np.nan
        )
        npar=estimator_rep.n_parameters
        n = X.shape[0]
        ncol = X.shape[1]
        ncomp=estimator_rep.n_components
        #degrees of freedom - binary variables        
        dof=(2**ncol)-((ncomp-1)+ncol*(2))
       
        stats_k = {'k':k,"LL": np.array(ll), "score": np.array(avg_ll),
            'aic':np.array(aic),'bic':np.array(bic),
            'caic':np.array(caic), 'sabic': np.array(sabic),
             'entropy': np.array(entropy),
             'relative_entropy': np.array(relentropy),
            'convergence':estimator_rep.converged_,
        'npar':npar,'n':n,'ncomp':ncomp, 'dof':dof
            }       
        stats.append(pd.DataFrame([stats_k]))
        obj_list.append(estimator_rep)
    
    stats=pd.concat(stats)
   # if verbose:
        #print("\nResults")
       # print(stats_k)
    
    # perform LRT ================
    nestedModelsLRT=list()
    for k in range(low+1,high):
        k_p=k-1
        ll_k=stats.loc[stats.ncomp==k,'LL'].values[0]
        ll_kp=stats.loc[stats.ncomp==k_p,'LL'].values[0]
        dof_k=stats.loc[stats.ncomp==k,'dof'][0]
        dof_kp=stats.loc[stats.ncomp==k_p,'dof'][0]
        a=lrt(ll_kp, ll_k, dof_kp,dof_k)
        a['k']=k
        nestedModelsLRT.append(a)
    nestedModelsLRT=pd.DataFrame(nestedModelsLRT)
    
    nestedModelsLRT=pd.merge(stats,nestedModelsLRT)
    print(nestedModelsLRT[['ncomp','npar','n','convergence','LL','dof','log-likelihood diff','dof diff', 'p value','aic','bic','entropy','relative_entropy']])
    
    # generate plots =================
    adicText=' (max_iterations ='+str(estimator_rep.max_iter) +')'
    plot_stats(stats, adicText)

    return stats ,obj_list

def plot_stats(stats, adicText=None):
    low=stats.ncomp.min()
    high=stats.ncomp.max()
    #TODO - add differential markers for converged x failed runs
    marker_mapping = {
        'True': 'o', # circle
        'False': 'x'  # square
    }
    # plot for log-likelihood
    fig, ax = plt.subplots()
    df_m=stats[['ncomp','LL']]
    ax.plot(df_m.ncomp, df_m[['LL']],'-',label='_log-likelihood')# 4. Add labels, legend, and display the plot
    ax.set_xlabel("k")
    ax.set_ylabel('log-likelihood')
    ax.set_title('log-likelihood'+adicText)
    plt.xticks(range(low, high,1))
    for category, marker_style in marker_mapping.items():
        # Select data points for the current category
        # Convert lists to numpy arrays for easier filtering
        x_arr = np.array(stats[['ncomp']])
        y_arr = np.array(stats[['LL']])
        categories = np.array([str(x[0]) for x in stats[['convergence']].values])
        cat_arr = np.array(categories)
        # Filter data
        x_cat = x_arr[cat_arr == category]
        y_cat = y_arr[cat_arr == category]
        
        # Plot the subset of data with the specific marker and add a label for the legend
        
        plt.scatter(x_cat, y_cat, marker=marker_style, label=f'Convergence {category}')
    plt.legend()
    plt.show()
    #bic sapib etc plots
    fig, ax = plt.subplots()

    
    for m in ['bic','aic','sabic','caic']:
        df_m=stats[['ncomp',m]]
        ax.plot(df_m.ncomp, df_m[[m]],'-', label=m,         
            markersize=2, )# 4. Add labels, legend, and display the plot
        ax.set_xlabel("k")
        ax.set_ylabel('metric')
        
        for category, marker_style in marker_mapping.items():
            # Select data points for the current category
            # Convert lists to numpy arrays for easier filtering
            x_arr = np.array(stats[['ncomp']])
            y_arr = np.array(stats[m])
            categories = np.array([str(x[0]) for x in stats[['convergence']].values])
            cat_arr = np.array(categories)
            # Filter data
            x_cat = x_arr[cat_arr == category]
            y_cat = y_arr[cat_arr == category]
            
            # Plot the subset of data with the specific marker and add a label for the legend
            
            plt.scatter(x_cat, y_cat, marker=marker_style, label=f'_Convergence {category}')
        ax.legend()
    plt.xticks(range(low, high,1))
    ax.set_title("Curve for obtained metrics"+adicText)
    plt.show()
    # plots for entropy
    fig, ax = plt.subplots()
    df_m=stats[['ncomp','entropy']]
    ax.plot(df_m.ncomp, df_m[['entropy']],'-',label='_entropy')# 4. Add labels, legend, and display the plot
    ax.set_xlabel("k")
    ax.set_ylabel('entropy')
    ax.set_title('entropy'+adicText)
    plt.xticks(range(low, high,1))
    for category, marker_style in marker_mapping.items():
        # Select data points for the current category
        # Convert lists to numpy arrays for easier filtering
        x_arr = np.array(stats[['ncomp']])
        y_arr = np.array(stats[['entropy']])
        categories = np.array([str(x[0]) for x in stats[['convergence']].values])
        cat_arr = np.array(categories)
        # Filter data
        x_cat = x_arr[cat_arr == category]
        y_cat = y_arr[cat_arr == category]
        
        # Plot the subset of data with the specific marker and add a label for the legend
        
        plt.scatter(x_cat, y_cat, marker=marker_style, label=f'Convergence {category}')
    plt.legend()
    plt.show()
    #relative entropy
    fig, ax = plt.subplots()
    df_m=stats[['ncomp','relative_entropy']]
    ax.plot(df_m.ncomp, df_m[['relative_entropy']],'-',label='_relative entropy')# 4. Add labels, legend, and display the plot
    ax.set_xlabel("k")
    ax.set_ylabel('relative entropy')
    ax.set_title('Relative Entropy'+adicText)
    plt.xticks(range(low, high,1))
    for category, marker_style in marker_mapping.items():
        x_arr = np.array(stats[['ncomp']])
        y_arr = np.array(stats[['relative_entropy']])
        categories = np.array([str(x[0]) for x in stats[['convergence']].values])
        cat_arr = np.array(categories)
        # Filter data
        x_cat = x_arr[cat_arr == category]
        y_cat = y_arr[cat_arr == category]
        
        # Plot the subset of data with the specific marker and add a label for the legend
        
        plt.scatter(x_cat, y_cat, marker=marker_style, label=f'Convergence {category}')
    plt.legend()
    plt.show()