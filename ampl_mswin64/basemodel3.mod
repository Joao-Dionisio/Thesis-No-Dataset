reset;
param Tmax;                     # upper bound on the number of periods
set components = {"joints", "oil", "winding"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price (of electricity?)
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param load_effect;                              # equipment load usage factor
param periodicity;					  # periodicity of preventive maintenance

var q{1..Tmax}>=0;                     # quantity produced/sold in period t
var qmax{1..Tmax}>=0;                  # adjusted capacity for period t
var rul{k in components,0..Tmax} >= 0, <= RULmax[k];   # remaining useful life for item k in components
var x{k in components,1..Tmax} binary;            # maintenance/replacement of parts k in components
var force_shutdown{1..Tmax} binary;

set failure_time;

# these are for testing
param best;
param index_best;

maximize profit:
    sum {t in 1..Tmax} (Price*q[t]-sum{k in components} Cost[k]*x[k,t]);


RUL0{k in components}: 
    rul[k,0] = RULmax[k];

#RUL{k in components, t in 1..Tmax}: 
#    rul[k,t] <=  rul[k,t-1]*d - load_effect*q[t]/Qmax + 2*RULmax[k]*x[k,t];

RUL1{t in 1..Tmax}: 
    rul["joints",t] <=  rul["joints",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["joints"]*x["joints",t];

RUL2{t in 1..Tmax}: 
    rul["oil",t] <=  rul["oil",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] + 3*RULmax["oil"]*x["oil",t]; 

RUL3{t in 1..Tmax}:     
    rul["winding",t] <= rul["winding",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] - load_effect*rul["oil",t]/RULmax["oil"] + 4*RULmax["winding"]*x["winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced


# adjust capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];

Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


# force smaller maintenances to occur when larger is made
oil_implies_joints{t in 1..Tmax}: 
	x["oil", t] <= x["joints", t];

winding_implies_oil{t in 1..Tmax}: 
	x["winding", t] <= x["oil",t]; # winding implies joints is above


everything_stops{t in 1..Tmax, k in components}: 
	rul[k,t] <= force_shutdown[t]*RULmax[k];

#everything_stops{t in 1..Tmax}: 
#	q[t] <= force_shutdown[t]*Qmax;


stopping_is_definitive{t in 1..Tmax-1}: 
	force_shutdown[t+1] <= force_shutdown[t];
	
	
# restrictions from amplpy
#stochastic_error{t in 1..Tmax, k in components}: 
#rul[k,t] <= rul[k,t]*failure_time[t];


# The following two restrictions may probably be removed

#Ends when winding reaches 0
#Early_stop{t in 1..Tmax}: 
#	sum{i in t..Tmax} (qmax[i] + sum{k in components} x[k,i]) <= rul["winding",t-1]*(Qmax+3)*Tmax; 


#Can't operate with 0 health on any component
#ensure_positive_rul {t in 1..Tmax, k in components}:
#	rul["winding",t]/RULmax[k] <= rul[k,t]; # ensures all components must be strictly positive, until we let the PT fail



#regular_maintenance{t in 1..Tmax, k in components}: 
#	x[k,t] = ceil((t mod periodicity/periodicity)); # doing maintenance every 10 years

data;
param:   RULmax   Cost :=
   joints	  10   1900
   oil  	  25   600000
   winding 	  50   1000000 # values from markovian model (in pounds)
;

param Tmax := 100;
param Price := 1;
param Qmax := 2000000;
param d := 0.9;
param load_effect := 1;
param best := 0;
param index_best := 0;

for {i in 1..1}{
#for {i in 1..Tmax}{
	let periodicity := i;

	option solver gurobi;
	# option solver knitro;
	option gurobi_options "timelim=10";
	solve;
	
	#display rul,x,qmax,q;
	#display {t in 1..Tmax} Price*q[t];
	#display {t in 1..Tmax} sum {k in components} Cost[k]*x[k,t];
	#display {t in 1..Tmax} force_shutdown[t]*Price*q[t];
	#display profit;
	if profit > best then {
		let best := profit;
		let index_best := i
	}
	display x;
}
display index_best,best;




