param Tmax := 30;                     # upper bound on the number of periods
set components = {"joints", "oil", "winding"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price (of electricity?)
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param load_effect;                              # equipment load usage factor

var q{1..Tmax}>=0;                     # quantity produced/sold in period t
var qmax{1..Tmax}>=0;                  # adjusted capacity for period t
var rul{k in components,0..Tmax} >= 0, <= RULmax[k];   # remaining useful life for item k in components
var x{k in components,1..Tmax} binary;            # maintenance/replacement of parts k in components
var force_shutdown{1..Tmax} binary;

maximize profit:
    sum {t in 1..Tmax} (Price*q[t]-sum{k in components} Cost[k]*x[k,t]);


RUL0{k in components}: 
    rul[k,0] = RULmax[k];

RUL1{t in 1..Tmax}: 
    rul["joints",t] <=  rul["joints",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["joints"]*x["joints",t];

RUL2{t in 1..Tmax}: 
    rul["oil",t] <=  rul["oil",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] + 3*RULmax["oil"]*x["oil",t]; 

RUL3{t in 1..Tmax}:     
    rul["winding",t] <= force_shutdown[t]*(rul["winding",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] - load_effect*rul["oil",t]/RULmax["oil"] + 4*RULmax["winding"]*x["winding",t] -1);


everything_stops{t in 1..Tmax, k in components}: rul[k,t] <= force_shutdown[t]*RULmax[k];
stopping_is_definitive{t in 1..Tmax-1}: force_shutdown[t+1] <= force_shutdown[t];


#Can't operate with 0 health
ensure_positive_rul {t in 1..Tmax, k in components}:
	rul["winding",t]/RULmax[k] <= rul[k,t]; # ensures all components must be strictly positive, until we let the PT fail

#Ends when winding reaches 0
Early_stop{t in 1..Tmax}: sum{i in t..Tmax} (qmax[i] + sum{k in components} x[k,i]) <= rul["winding",t-1]*(Qmax+3)*Tmax; 


# adjust capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];

Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


# force smaller maintenances to occur when larger is made
oil_implies_joints{t in 1..Tmax}: x["oil", t] <= x["joints", t];

winding_implies_oil{t in 1..Tmax}: x["winding", t] <= x["oil",t]; # winding implies joints is above


data;
param:   RULmax   Cost :=
   joints	  10   250
   oil  	  25   500
   winding 	  50   5000
;

param Price := 1;
param Qmax := 10000;
param d := 0.9;
param load_effect := 1;


option solver gurobi;
solve;

display rul,x,qmax,q;
display {t in 1..Tmax} Price*q[t];
display {t in 1..Tmax} sum {k in components} Cost[k]*x[k,t];
display {t in 1..Tmax} force_shutdown[t]*Price*q[t];

display profit;
reset;