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

Software: Python 3.7, GAMS 33.1, R 4.0

License: GAMS, Gurobi, BARON, MOSEK

Python Package: IPython 7.19.0, pandas 1.1.5, numpy 1.19.2, scipy 1.5.2, gurobipy 9.1.0, mosek 9.2.49, sklearn 0.23.2, pyitlib 0.2.2, replicate 0.2.2, matplotlib 3.3.1

R Library: Openxlsx 4.2.4, gplots 3.1.1, RColorBrewer 1.1

(R was used only for visualization.)

## Notes

A Dockerfile comes with Python, necessary packages, GAMS and a demo license of GAMS. However, to fully replicate codes, a user needs to include their own licenses of GAMS, Gurobi, BARON, and MOSEK. Gurobi and MOSEK provide free academic licenses. Yet, GAMS and BARON do not.



