import pdb
import pandas as pd
import numpy as np
import scipy.stats as ss


def import_data(incsv):
    X_df = pd.read_csv(incsv, header=0)
    return X_df


def fit_gmm(mod: dict, y):
    """Fit the GMM using EM algorithm"""

    def update_tau():
        tau_hat = mod['tau'][:]
        for i in range(len(y)):
            tau_hat[i] = np.log(mod['p']) + ss.norm.logpdf(y[i], loc=mod['m'],
                                                           scale=mod['cov']**0.5)
            tau_hat[i] = np.exp(tau_hat[i])
            tau_hat[i] = tau_hat[i] / sum(tau_hat[i])
        return tau_hat

    def update_m():
        return np.dot(mod['tau'].T, y) / np.sum(mod['tau'], axis=0)

    def update_cov(mu):
        temp_cov = np.zeros(3)
        for k in range(3):
            temp_cov[k] = np.cov(mod['tau'].T[k]*(y - mu[k]).T)/np.sum(mod['tau'].T[k])
        return temp_cov

    def update_p():
        return np.mean(mod['tau'], axis=0)

    # Main EM iteration
    for iter in range(5):
        mod['tau'] = update_tau()
        mod['m'] = update_m()
        mod['cov'] = update_cov(mod['m'])
        mod['p'] = update_p()

    return mod


def export_model(mod, outcsv):
    z = np.argmax(mod['tau'], axis=1)
    m = mod['m']
    p = mod['p']
    cov = mod['cov']

    mod_df = pd.concat([pd.DataFrame({'param': ["z_%03d" % i for i in range(1, len(z)+1)],
                                      'value': z.astype('int')}),
                        pd.DataFrame({'param': ["m_%01d" % i for i in range(1, len(m)+1)],
                                      'value': m}),
                        pd.DataFrame({'param': ["cov_%01d" % i for i in range(1, len(cov)+1)],
                                      'value': cov}),
                        pd.DataFrame({'param': ["p_%01d" % i for i in range(1, len(p)+1)],
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
    mod['m'] = np.array(mod['m'])
    mod['p'] = np.array(mod['p'])
    mod['cov'] = np.array(mod['cov'])

    return mod


def gmm_iris_1d(incsv, kmeanscsv, outcsv):
    X_df = import_data(incsv)
    x = np.array(X_df['x'])

    mod = init_kmeans(kmeanscsv)
    mod = fit_gmm(mod, x)
    export_model(mod, outcsv)


gmm_iris_1d("data/iris-1d.csv","results/kmeans_iris-1d.csv",
         "results/em_iris-1d.csv")
