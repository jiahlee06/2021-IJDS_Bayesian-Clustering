import pdb
import pandas as pd
from pandas import DataFrame
import numpy as np
import scipy.stats as ss
import itertools as it


def import_data(incsv):
    X_df = pd.read_csv(incsv, header=0)
    return X_df

def fit_gmm(mod: dict, y):
    """Fit the GMM using EM algorithm"""

    K = 5
    d = 4

    def update_tau():
        tau_hat = mod['tau'][:]
        for i in range(len(y)):
            for k in range(K):
                tau_hat[i, k] = np.log(
                    mod['p'][k]) + ss.multivariate_normal.logpdf(y[i], mean=mod['m'][k], cov=mod['cov'][k])
            tau_hat[i] = np.exp(tau_hat[i])
            tau_hat[i] = tau_hat[i] / sum(tau_hat[i])
        return tau_hat

    def update_m():
        return np.dot(mod['tau'].T, y) / np.repeat(np.sum(mod['tau'],
                                                          axis=0), d).reshape((K, d))
    def update_cov(mu):
        temp_cov = np.zeros((K,d,d))
        for k in range(K):
            temp_cov[k,:] = np.dot(mod['tau'].T[k]*(y - mu[k]).T, (y - mu[k]))/np.sum(mod['tau'].T[k])
        return temp_cov
    
    def update_p():
        return np.mean(mod['tau'], axis=0)

    # Main EM iteration
    for iter in range(50):
        mod['tau'] = update_tau()
        mod['m'] = update_m()
        mod['cov'] = update_cov(mod['m'])
        mod['p'] = update_p()

    return mod


def export_model(mod, outcsv):
    z = np.argmax(mod['tau'], axis=1)
    m = mod['m']
    cov = mod['cov']
    p = mod['p']

    K = 5
    d = 4

    mod_df = pd.concat([DataFrame({'param': ["z_%03d" % i for i in range(1, len(z)+1)],
                                   'value': z.astype('int')}),
                        DataFrame({'param': ["m_%01d-%01d" % (k, d) for (k, d) in it.product(range(1, K+1), range(1, d+1))],
                                   'value': m.reshape(-1)}),
                        DataFrame({'param': ["cov_%01d-%01d-%01d" % (k, d1, d2) for (k, d1, d2) in it.product(range(1, K+1), range(1, d+1), range(1, d+1))],
                                   'value': cov.reshape(-1)}),
                        DataFrame({'param': ["p_%01d" % i for i in range(1, len(p)+1)],
                                   'value': p})],
                       axis=0, ignore_index=True)
    mod_df.to_csv(outcsv, header=True, index=False)


def init_kmeans(kmeanscsv):
    """Initialize the GMM model using the output of kmeans"""
    mod = {'z': [], 'm': [], 'p': [], 'cov': []}
    mod_df = pd.read_csv(kmeanscsv, header=0)

    for index, row in mod_df.iterrows():
        param, i_str = row[0].split("_")
        eval("mod['{}'].append({})".format(param, row[1]))

    mod['z'] = np.array(mod['z'], dtype=int)
    mod['tau'] = pd.get_dummies(pd.Series(mod['z'])).values.astype(float)
    mod['m'] = np.array(mod['m']).reshape((5,4))
    mod['p'] = np.array(mod['p'])
    mod['cov'] = np.array(mod['cov']).reshape((5,4,4))

    return mod


def gmm_pam50_4d(incsv, kmeanscsv, outcsv):
    X_df = import_data(incsv)
    cols = ['component1',
            'component2', 'component3', 'component4']
    x = np.array(X_df[cols])

    mod = init_kmeans(kmeanscsv)
    mod = fit_gmm(mod, x)
    export_model(mod, outcsv)


gmm_pam50_4d("processed/pam50-4d.csv", "model/kmeans_pam50-4d.csv", "model/em_pam50-4d.csv")
