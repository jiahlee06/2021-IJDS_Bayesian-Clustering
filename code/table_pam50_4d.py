import pdb
import re
import pandas as pd
import numpy as np
import itertools as it

from pyitlib import discrete_random_variable as drv

pam50_4d_files = ["processed/pam50-4d.csv", "model/kmeans_pam50-4d.csv", "model/em_pam50-4d.csv", "model/minlp_pam50-4d.csv", "model/bandi_pam50-4d.csv",
                "report/metric-table_pam50-4d.csv", "report/metric-table_pam50-4d.tex"]

def dist_em(z_true, z_hat):
    """ Compute the error in z"""

    z_true = pd.get_dummies(z_true).astype(int)
    z_hat = pd.get_dummies(z_hat).astype(int)
    z_err = np.linalg.norm(z_hat - z_true, ord=np.inf, axis=1)

    return np.sum(z_err)/len(z_err)

def dist_vi(z1, z2):
    """ Compute the error in z"""

    vi = drv.information_variation(z1, z2, base=2)
    ubd = 2 * np.log(3)

    return vi/ubd


def permute_model(theta_true, mod):

    # Get the best permutation of components
    best_d = np.inf
    best_perm = np.array(5)
    for perm in it.permutations(range(5)):
        perm = np.array(perm)
        m_hat_p = mod['m'][perm]
        d = np.linalg.norm(m_hat_p - theta_true['m'])
        if d < best_d:
            best_d = d
            best_perm = perm

    # Apply permutation to model parameters
    mod['m'] = mod['m'][best_perm]
    z_p = mod['z']
    perm_mat = np.zeros((len(best_perm), len(best_perm)))
    for idx, i in enumerate(best_perm):
        perm_mat[idx, i] = 1
    z_p = np.dot(np.asarray(pd.get_dummies(z_p)), perm_mat.T)
    mod['z'] = np.argmax(z_p, axis=1)

    return mod


def import_model(modelcsv):
    """Read model csv into pandas data frame"""

    n = 232
    K = 5
    d = 4

    mod = pd.read_csv(modelcsv)
    z = mod['value'][0:n].values.astype('int')
    m = mod['value'][n:(n+K*d)].values.astype('float')
    m = np.reshape(m, (K, d))
  
    match = re.search('model/(.+?)_', modelcsv)
    mod = {'source': match.group(1), 'z': z, 'm': m}

    return mod


def make_metric_table(datacsv, modelcsvlist, outputlist):

    # import permutated models
    models = []
    for modelcsv in modelcsvlist:
        print(modelcsv)
        mod = import_model(modelcsv)
        if 'em' in modelcsv or 'minlp' in modelcsv or 'bandi' in modelcsv:
            mod = permute_model(models[0], mod)
        models.append(mod)

    # record distances
    dist = []
    for (i, j) in it.product(range(len(models)), range(len(models))):
        if i==j:
            dist.append(0)
        elif i<j:
            dist.append(dist_em(models[i]['z'], models[j]['z']))
        else: 
            dist.append(dist_vi(models[i]['z'], models[j]['z']))

    # Combine metrics into a data frame
    cols = ['Kmeans', 'EM', 'MINLP', 'Bandi']
    dist_df = pd.DataFrame(columns = cols)
    for k in range(len(cols)):
        dist_df[cols[k]] = dist[len(cols)*k:len(cols)*(k+1)]
    dist_df.index = cols
    dist_df = dist_df.transpose()
    print(dist_df)
  
    # Export data frame
    dist_df.to_csv(outputlist[0], float_format='%0.3f')
    dist_df.to_latex(outputlist[1], float_format='%0.3f')


if __name__ == "__main__":
    make_metric_table(pam50_4d_files[0], pam50_4d_files[1:5], pam50_4d_files[5:7])



    
