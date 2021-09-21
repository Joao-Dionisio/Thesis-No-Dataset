param Tmax := 40;                     # upper bound on the number of periods
set components = {"oil", "paper", "copper"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param c;                              # equipment load usage factor

var q{1..Tmax} >= 0;                     # quantity produced/sold in period t
var qmax{1..Tmax} >= 0;                  # adjusted capacity for period t
var rul{k in components,0..Tmax} >=0, <= RULmax[k];   # remaining useful life for item k in K
var x{k in components,1..Tmax} binary;            # maintenance/replacement of parts k in K


maximize profit:
    sum {t in 1..Tmax} (Price*q[t] - sum {k in components} Cost[k]*x[k,t]);


RUL0{k in components}: 
    rul[k,0] = 0;

RUL{k in components, t in 1..Tmax}: 
    rul[k,t] <=  rul[k,t-1]*d - c*q[t]/Qmax + 2*RULmax[k]*x[k,t];

# adjust capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];
Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


data;
param:   RULmax   Cost :=
   oil	  10   2500
   paper  25   5000
   copper 50   5000000
;

param Price := 1;
param Qmax := 10000;
param d := 0.9;
param c := 1;


option solver gurobi;
# option solver knitro;
solve;

#display rul,x,qmax,q;
#display {t in 1..Tmax} Price*q[t];
#display {t in 1..Tmax} sum {k in components} Cost[k]*x[k,t];
# display R;
display profit;
reset;
