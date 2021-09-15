$ontext

  Maximum a-posteriori Gaussian mixture model based on a hierarchical model.
  Iterates between a step to estimate assignments and a step to estimate hyperparameters
  The first step in GAMS and the second step in Python.

  Edited by Ji Ah Lee, 2021

$offtext

* suppress input file listing
$offlisting
* suppress equation listing
option limrow = 0;
* suppress variable listing
option limcol = 0;
* suppress the solution listing -- we'll display our own and output to gdx
option solprint = off ;

Sets
    i         samples
    k         components
    d         features ;
alias (d, d1, d2);

Parameters
    y(i,d)               "observed data"
    det(k)		'determinant of precision  matrices'
    a(k)		'hyperparameter from prior of proportions'
    m(k,d)		'location parameter from prior of component means'
    sigma_inv(k,d1,d2)	'sampling precision matrices'
    c1(k)
    c2(k)
    c3(k)
;

Scalars 
    n 'total samples'
    nk 'total components' 
    nd 'total dimensions' ;


$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$load i k d y sigma_inv det a m c1 c2 c3
$gdxin

n = card(i);
nk = card(k);
nd = card(d);


Variables
    z(i,k)   'component assignments'
    s(k)     'count of data points allocated to component'
    f        'objective function value'
;

Binary Variable z(i,k) ;
Integer Variable s(k) ;

z.up(i,k) = 1;
s.lo(k) = 2;
s.up(k) = n;

Equations
    propz(k)       'count of data assigned to component'
    assignz(i)     'sum_k z(i,k) = 1'
    obj            'objective function'
;

propz(k)..       s(k) =e= sum(i, z(i,k));

assignz(i)..     sum(k, z(i,k)) =e= 1;

obj..            f =e= sum((i,k), z(i,k) * (-log(c1(k)) -log(det(k))/2 - c3(k) * log(1 + c2(k) * sum((d1,d2), (y(i, d1) - m(k, d1)) * sigma_inv(k,d1,d2) * (y(i, d2) - m(k, d2)))))) + sum(k, a(k)*log(a(k))-a(k)-log(a(k))/2+log(2*pi)/2+1/(12*a(k))-1/(360*power(a(k),3))+1/(1260*(power(a(k),5))) -(s(k)+a(k))*log(s(k)+a(k))+s(k)+a(k)+log(s(k)+a(k))/2-log(2*pi)/2-1/(12*(s(k)+a(k)))+1/(360*power(s(k)+a(k),3))-1/(1260*power(s(k)+a(k),5)));

* sum(k, a(k)*log(a(k))-a(k)-log(a(k))/2+log(2*pi)/2+1/(12*a(k))-1/(360*power(a(k),3))+1/(1260*(power(a(k),5))) -(s(k)+a(k))*log(s(k)+a(k))+s(k)+a(k)+log(s(k)+a(k))/2-log(2*pi)/2-1/(12*(s(k)+a(k)))+1/(360*power(s(k)+a(k),3))-1/(1260*power(s(k)+a(k),5)));

* sum(k, logGamma(a) - logGamma(s(k)+a));

Model gmm /all/ ;

* Solve gmm minimizing f using minlp ;
* Display f.l, z.l, m.l, p.l ;
