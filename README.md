# Bayesian Clustering via MINLP for GMM

A short description of how to run codes and what is needed.

## To run codes

To run K-means, EM, MIO from Bandi's and metric's calculations, you can run files starting with 'kmeans_', em_', 'bandi_', and 'table_' respectively.

To run MINLP for standard data sets, you can run via

	python minlp_em.py --file_name=iris-4d

replacing iris-4d with brca, iris-1d, wine. To run MINLP for pam50-4d,

	python minlp_pam50.py

will do.

## What is needed

Software: Python 3.7, R 4.0
License: GAMS, Gurobi, BARON, MOSEK

Python Package: pdb, pandas, numpy, scipy, itertools, gurobipy, mosek, sklearn, IPython, re, pyitlib, gams, os, sys, argparse, replicate, matplotlib, random, csv
R Library: Openxlsx, gplots, RColorBrewer



