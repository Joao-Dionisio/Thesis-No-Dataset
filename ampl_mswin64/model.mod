param Tmax;                     # upper bound on the number of periods
set components = {"joints", "oil", "winding"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price (of electricity?)
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param load_effect;                              # equipment load usage factor
param periodicity;					  # periodicity of preventive maintenance
set failure_time;

# for testing, not actually used here
param best;
param index_best;

var q{1..Tmax}>=0;                     # quantity produced/sold in period t
var qmax{1..Tmax}>=0;                  # adjusted capacity for period t
var rul{k in components,0..Tmax} >= 0, <= RULmax[k];   # remaining useful life for item k in components
var x{k in components,1..Tmax} binary;            # maintenance/replacement of parts k in components
var force_shutdown{1..Tmax} binary;

maximize profit:
    sum {t in 1..Tmax} (Price*q[t]-sum{k in components} Cost[k]*x[k,t]);

# Assuming PT is brand new
RUL0{k in components}: 
    #rul[k,0] = RULmax[k]; 
    rul[k,0] = 0; # <- takes into consideration if it's worth it to buy an equipement

RUL1{t in 1..Tmax}: 
    rul["joints",t] <=  rul["joints",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["joints"]*x["joints",t];

RUL2{t in 1..Tmax}: 
    rul["oil",t] <=  rul["oil",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] + 3*RULmax["oil"]*x["oil",t]; 

RUL3{t in 1..Tmax}:     
    rul["winding",t] <= rul["winding",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] - load_effect*rul["oil",t]/RULmax["oil"] + 4*RULmax["winding"]*x["winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced


# from paper
# DP = log(2FAL/(MP*78.98-23.3)/-0.008    <- this relates moisture to DP. Maybe get relation between oil condition and moisture?
# 2FAL = (MP*78.98-23.3)*exp(-0.008PD)


# Adjusting capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];

Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


# Forcing smaller maintenances to occur when larger is made
oil_implies_joints{t in 1..Tmax}: 
	x["oil", t] <= x["joints", t];

winding_implies_oil{t in 1..Tmax}: 
	x["winding", t] <= x["oil",t]; # winding implies joints is above


everything_stops{t in 1..Tmax, k in components}: 
	rul[k,t] <= force_shutdown[t]*RULmax[k];


stopping_is_definitive{t in 1..Tmax-1}: 
	force_shutdown[t+1] <= force_shutdown[t];
	
	
# Failures provided by GP
# gp_failure{t in failures, k in components} : x[k,t-1] = 1;
