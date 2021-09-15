from pandas import DataFrame, read_csv, concat
import sklearn.cluster as sc
import numpy as np
import pdb


def import_data(incsv):
    X_df = read_csv(incsv, header=0)
    return X_df


def fit_kmeans(X_df):
    X = np.array(X_df['x'])
    kmeans = sc.KMeans(n_clusters=3, init='random').fit(np.array([X, ]).T)
    cov = np.zeros(3)
    for k in range(3):
        cov[k] = np.cov(np.array([X,]).T[kmeans.labels_==k,:].T)
    return kmeans, cov


def export_model(mod, outcsv, cov):
    z = mod.labels_
    m = mod.cluster_centers_
    k, p = np.unique(z, return_counts=True)
    p = p / len(z)

    mod_df = concat([DataFrame({'param': ["z_%03d" % i for i in range(1, len(z)+1)],
                                'value': z.astype('int')}),
                     DataFrame({'param': ["m_%01d" % i for i in range(1, len(m)+1)],
                                'value': m[:, 0]}),
                     DataFrame({'param': ["cov_%01d" % i for i in range(1, len(cov)+1)],
                                'value': cov}),
                     DataFrame({'param': ["p_%01d" % i for i in range(1, len(p)+1)],
                                'value': p})],
                    axis=0, ignore_index=True)
    mod_df.to_csv(outcsv, header=True, index=False)


def kmeans_iris_1d(incsv, outcsv):
    X_df = import_data(incsv)
    mod, cov = fit_kmeans(X_df)
    export_model(mod, outcsv, cov)


kmeans_iris_1d("processed/iris-1d.csv","model/kmeans_iris-1d.csv")
