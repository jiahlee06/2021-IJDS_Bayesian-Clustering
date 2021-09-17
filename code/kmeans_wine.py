from pandas import DataFrame, read_csv, concat
import sklearn.cluster as sc
import numpy as np
import itertools as it
import pdb


def import_data(incsv):
    X_df = read_csv(incsv, header=0)
    return X_df


def fit_kmeans(X_df):
    cols = ['alcohol', 'malic_acid', 'ash', 'ash_alkalinity', 'magnesium', 'total_phenols',
            'flavanoids', 'nonflavanoid_penols', 'proanthocyanins', 'color_intensity', 'hue', 'od', 'proline']
    X = np.array(X_df[cols])
    kmeans = sc.KMeans(n_clusters=3, init='random').fit(np.array(X))
    cov = np.zeros((3, 13, 13))
    for k in range(3):
        cov[k,:] = np.cov(np.array(X)[kmeans.labels_==k,:].T)
    return kmeans, cov


def export_model(mod, outcsv, cov):
    z = mod.labels_
    m = mod.cluster_centers_
    k, p = np.unique(z, return_counts=True)
    p = p / len(z)

    K = 3
    d = 13

    mod_df = concat([DataFrame({'param': ["z_%03d" % i for i in range(1, len(z)+1)],
                                'value': z.astype('int')}),
                     DataFrame({'param': ["m_%01d-%01d" % (k, d) for (k, d) in it.product(range(1, K+1), range(1, d+1))],
                                'value': m.reshape(-1)}),
                     DataFrame({'param': ["cov_%01d-%01d-%01d" % (k, d1, d2) for (k, d1, d2) in it.product(range(1, K+1), range(1, d+1), range(1, d+1))],
                                'value': cov.reshape(-1)}),
                      DataFrame({'param': ["p_%01d" % i for i in range(1, len(p)+1)],
                                'value': p})],
                    axis=0, ignore_index=True)
    mod_df.to_csv(outcsv, header=True, index=False)


def kmeans_wine(incsv, outcsv):
    X_df = import_data(incsv)
    mod, cov = fit_kmeans(X_df)
    export_model(mod, outcsv, cov)


kmeans_wine("data/wine.csv", "results/kmeans_wine.csv")
