import gams
import os
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
import itertools as it
import sklearn.covariance as sklc
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import argparse
import replicate

import random
import scipy
import csv

# Load all data into python
# Add Model from GMS file

path = os.getcwd()

def add_data_to_db(ws, features, samples, X, file_name):
    
    # create database in workspace
    db = ws.add_database()
    
    i = db.add_set("i", 1, "samples")
    for s in samples:
        i.add_record(s)

    if 'pam' in file_name:
        norm_i = db.add_set_dc("norm_i", [i], "normal samples")
        for t in range(1,12+1):
            norm_i.add_record("i"+str(t))
            
        basal_i = db.add_set_dc("basal_i", [i], "basal samples")
        for t in range(1+12,69+1):
            basal_i.add_record("i"+str(t))
            
        her2_i = db.add_set_dc("her2_i", [i], "her2 samples")
        for t in range(1+69,104+1):
            her2_i.add_record("i"+str(t))      
            
        lumA_i = db.add_set_dc("lumA_i", [i], "lumA samples")
        for t in range(1+104,127+1):
            lumA_i.add_record("i"+str(t))
            
        lumB_i = db.add_set_dc("lumB_i", [i], "lumB samples")
        for t in range(1+127,139+1):
            lumB_i.add_record("i"+str(t))
            
        
    d = db.add_set("d", 1, "features")
    for f in features:
        d.add_record(f)
    
    y = db.add_parameter_dc("y", [i,d], "observed data")
    for k, v in iter(X.items()):
        y.add_record(k).value = v
        
    return db
    
    
def read_csv_data(filename):
    ds = pd.read_csv(filename, index_col=0)
    ds = ds.drop(columns=["class"], errors = 'ignore')
    ds = ds.drop(columns=["diagnosis"], errors = 'ignore')
    ds.index = ["i{}".format(i) for i in ds.index]
    
    features = ds.columns
    samples = ds.index
    X = ds.stack().to_dict()
    
    return (features, samples, X)
            

def main(file_name, model, time_per_iter):
    
    if 'iris' in file_name:
        K = 3
    elif 'wine' in file_name:
        K = 3
    elif 'brca' in file_name:
        K = 2
    elif 'pam50-4d' in file_name:
        K = 5
    else:
        K = 50
  
  
    ws = gams.GamsWorkspace(system_directory = "/opt/gams/gams33.1_linux_x64_64_sfx/",
	debug = gams.DebugLevel.ShowLog )

    cp = ws.add_checkpoint()
    
    # initialize a GAMSCheckpoint by running a GAMSJob
    gmm = ws.add_job_from_file(path+"/"+model+".gms")
    
    # populate the problem database
    features,samples,X = read_csv_data("data/"+file_name+".csv")
    db = add_data_to_db(ws, features, samples, X, file_name)

    k = db.add_set("k", 1, "clusters")
    for i in range(K):
        k.add_record("k{}".format(i+1))

    d1 = db.get_set("d")
    d2 = db.get_set("d")
    k = db.get_set("k")

    a_value = len(samples)/K
    D = len(features)   
 
    # sigma_inv, m
    sigma_inv_np = np.zeros((K, D, D))
    m_np = np.zeros((K, D))
    for k in range(K):
        sigma_inv_np[k,:,:] = np.eye(D)

    sigma_inv = db.add_parameter("sigma_inv", 3, "sampling precison matrices")
    m = db.add_parameter("m", 2, "hyperparameter from prior of proportions")
    for k,d1,d2 in it.product(range(K), range(D), range(D)):
        key = ("k{}".format(k+1), features[d1], features[d2])
        sigma_inv.add_record(key).value = sigma_inv_np[k,d1,d2]
    for k,d in it.product(range(K), range(D)):
        key = ("k{}".format(k+1), features[d])
        m.add_record(key).value = m_np[k,d]

    # det, a
    det_sigma_inv = np.ones(K)
    a_np = a_value*np.ones(K)
    c1_np = np.ones(K)
    c2_np = np.ones(K)
    c3_np = np.ones(K)
    det = db.add_parameter("det", 1, "determinant of precison matrices")
    a = db.add_parameter("a", 1, "hyperparameter from prior of proportions")
    c1 = db.add_parameter("c1", 1)
    c2 = db.add_parameter("c2", 1)
    c3 = db.add_parameter("c3", 1)
    for k in range(K):
        key = ("k{}".format(k+1))
        det.add_record(key).value = det_sigma_inv[k] 
        a.add_record(key).value = a_np[k]
        c1.add_record(key).value = c1_np[k]
        c2.add_record(key).value = c2_np[k]
        c3.add_record(key).value = c3_np[k]

    opt = ws.add_options()
    opt.defines["gdxincname"] = db.name
    opt.all_model_types = 'baron'
    opt.etlim = time_per_iter

    gmm.run(opt, checkpoint=cp, databases = db)
    
    
    # create a GAMSModelInstance with a placeholder for sigma_inv
    # instantiate the GAMSModelInstance and pass a model definition and GAMSModifier to declare bmult mutable
    model_statement = "gmm use " + model[0:5] + " min f"

    mi = cp.add_modelinstance()
    sigma_inv = mi.sync_db.add_parameter("sigma_inv", 3, "sampling precison matrices")
    det = mi.sync_db.add_parameter("det", 1, "determinant of precison matrices")
    a = mi.sync_db.add_parameter("a", 1, "hyperparameter from prior of proportions")
    m = mi.sync_db.add_parameter("m", 2, "location parameter from prior of component menas")
    c1 = mi.sync_db.add_parameter("c1", 1)
    c2 = mi.sync_db.add_parameter("c2", 1)
    c3 = mi.sync_db.add_parameter("c3", 1)

    mi.instantiate(model_statement, [gams.GamsModifier(sigma_inv), gams.GamsModifier(det), gams.GamsModifier(a), gams.GamsModifier(m), gams.GamsModifier(c1), gams.GamsModifier(c2), gams.GamsModifier(c3)], opt)

    # Initialize the model usingthe output of EM
    mod = {'z': [], 'm': [], 'p': [], 'cov': []}
    mod_df = pd.read_csv("results/em_"+file_name+".csv", header=0)

    for index, row in mod_df.iterrows():
        param, i_str = row[0].split("_")
        eval("mod['{}'].append({})".format(param, row[1]))
    mod['z'] = np.array(mod['z'], dtype=int)
   
    ds = pd.read_csv("data/"+file_name+".csv", index_col=0)
    ds = ds.drop(columns=["class"], errors = 'ignore')
    ds = ds.drop(columns=["diagnosis"], errors = 'ignore')
    ds = ds.values

    for k in range(K):
        if len(features) > 1:
            sigma_inv_np[k,:,:] = np.linalg.pinv(np.cov(ds[mod['z']==k,:].T))
            det_sigma_inv[k] = np.linalg.det(sigma_inv_np[k,:,:])
        else:
            sigma_inv_np[k,:,:] = 1/np.cov(ds[mod['z']==k,:].T)
            det_sigma_inv[k] = sigma_inv_np[k,:,:]
        m_np[k,:] = np.mean(ds[mod['z']==k,:], axis = 0)
        a_np[k] = a_value*K*sum(mod['z']==k)/len(mod['z'])

    # bootstrap
    kappa = np.zeros(K)
    nu = np.zeros(K)
    bootstrap_B = 300
    for k in range(K):
        ds_temp = ds[mod['z']==k]
        m_bootstrap = np.zeros((bootstrap_B, D))
        sigma_bootstrap = np.zeros((bootstrap_B, D, D))
        for i in range(bootstrap_B):
            temp_index = np.random.choice(range(len(ds_temp)), len(ds_temp))
            m_bootstrap[i,:] = np.mean(ds_temp[temp_index], axis = 0)
            sigma_bootstrap[i,:] = np.cov(ds_temp[temp_index].T)

        if D > 1:
            kappa[k] = np.linalg.norm(np.cov(ds[mod['z']==k,:].T) @ np.linalg.pinv(np.cov(m_bootstrap.T)))
            norm_temp = np.linalg.norm(np.cov(ds[mod['z']==k,:].T) @ np.linalg.pinv(sigma_bootstrap.mean(axis=0)))

        else:
            kappa[k] = np.cov(ds[mod['z']==k,:].T) / np.cov(m_bootstrap.T)
            norm_temp = np.cov(ds[mod['z']==k,:].T) / np.mean(sigma_bootstrap)
        print(norm_temp)
        nu[k] = max((D+1) * (norm_temp + 1) / (1 - norm_temp), D+2)
        print((D+1) * (norm_temp + 1) / (1 - norm_temp))
        print(kappa[k])

        c1_np[k] = scipy.special.gamma((nu[k]+1)/2) / (scipy.special.gamma((nu[k]-D+1)/2) * (nu[k]-D+1)**(D/2) * ((kappa[k]+1)*(nu[k]+D+1)/(kappa[k]*(nu[k]-D+1)))**(D/2))
        c2_np[k] = kappa[k] / ((kappa[k]+1)*(nu[k]+D+1))
        c3_np[k] = (-nu[k]-1)/2


    # put initial values in
    for k,d1,d2 in it.product(range(K), range(D), range(D)):
        key = ("k{}".format(k+1), features[d1], features[d2])
        sigma_inv.add_record(key).value = sigma_inv_np[k,d1,d2]

    for k,d in it.product(range(K), range(D)):
        key = ("k{}".format(k+1), features[d])
        m.add_record(key).value = m_np[k,d]

    for k in range(K):
        key = ("k{}".format(k+1))
        det.add_record(key).value = det_sigma_inv[k]
        a.add_record(key).value = a_np[k]
        c1.add_record(key).value = c1_np[k]
        c2.add_record(key).value = c2_np[k]
        c3.add_record(key).value = c3_np[k]

    mi.solve()

    # record output
    z_np = np.zeros((len(samples),K))
    for k_np,k in enumerate(gmm.out_db["k"]):
        for i_np,i in enumerate(gmm.out_db["i"]):
            z_key = (i.key(0), k.key(0))
            z_np[i_np,k_np] = mi.sync_db["z"].find_record(z_key).level 
    f = mi.sync_db.get_variable("f").find_record().level

    z = np.zeros(len(samples))
    m = np.zeros((K, D)) 
    cov = np.zeros((K, D, D))
    p = np.zeros(K)
    for i in range(len(samples)):
        z[i] = round(np.sum(z_np[i] * np.array(range(K))))

    for k in range(K):
        m[k,:] = np.mean(ds[z==k,:], axis=0)
        cov[k,:,:] = np.cov(ds[z==k,:].T)
        p[k] = sum(z==k)/len(samples)
    
    mod_df = pd.concat([DataFrame({'param': ["z_%03d" % i for i in range(1, len(z)+1)],
                                   'value': z.astype('int')}),
                        DataFrame({'param': ["m_%01d-%01d" % (k, d) for (k, d) in it.product(range(1, K+1), range(1, D+1))],
                                   'value': m.reshape(-1)}),
                        DataFrame({'param': ["cov_%01d-%01d-%01d" % (k, d1, d2) for (k, d1, d2) in it.product(range(1, K+1), range(1, D+1), range(1, D+1))],
                                   'value': cov.reshape(-1)}),
                        DataFrame({'param': ["p_%01d" % i for i in range(1, len(p)+1)],
                                   'value': p}),
                        DataFrame({'param': "f",
                                   'value': np.array([f])})],
                       axis=0, ignore_index=True)
    mod_df.to_csv("results/minlp_"+file_name+".csv", header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type = str, default = "iris-4d")
    parser.add_argument("--model", type = str, default = "minlp-bayesian")
    parser.add_argument("--time_per_iter", type = int, default = 360)
    args = parser.parse_args()

    main(args.file_name, args.model, args.time_per_iter)


