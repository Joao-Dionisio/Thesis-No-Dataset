# ok vê o email do helder, provavelmente é útil

param Tmax;                     # upper bound on the number of periods
set components = {"OPS", "Cooling System", "Oil", "Winding"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price (of electricity?)
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param load_effect;                              # equipment load usage factor
param periodicity;					  # periodicity of preventive maintenance

param r = 1000; # these are dependant on PT type (type of cooling system and whatnot)
param x = 0.8;
param k_11 = 0.5;
param tau_0 = 210;


var A; # = 1.367*10^10*(RHoil)^2 + 8.167*10^10*RHoil - 3.25*10^9
var RHoil{1..Tmax}>= 0; #=WC[t]/OilSat23
var WC{1..Tmax}>=0; # = moist*2.24*e^(-0.04*Tsample)



param oilsat23 = 55; # from helder's email
param EA = 12800;

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
    rul["OPS",t] <=  rul["OPS",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["OPS"]*x["OPS",t];

RUL2{t in 1..Tmax}: 
    rul["Cooling System,t] <=  rul["Cooling System",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] + 3*RULmax["Cooling System"]*x["Cooling System",t]; 

RUL3{t in 1..Tmax}: 
    rul["Oil,t] <=  rul["Oil",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] + 3*RULmax["Oil"]*x["Oil",t]; 

RUL4{t in 1..Tmax}:     
    rul["Winding",t] <= rul["Winding",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] - load_effect*rul["oil",t]/RULmax["oil"] + 4*RULmax["Winding"]*x["Winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced

    #rul["Winding",t] <= rul["Winding",t-1]*d - 2^((temperature[t]-98)/6)*something_with_moisture+ 4*RULmax["Winding"]*x["Winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced
    #rul["Winding",t] <= 1/((1/rul['Winding',t-1])+(A[i]*(t[i]-t[i-1]))/(3600*e^(E_A/(8.314*(Thsi+273.15))))) + 4*RULmax["Winding"]*x["Winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced

#temperature{t in 1..Tmax}:
    temperature[t] = temperature[t-1] + (((1+r*(q^2))/(1+r))^x)*(1-e^((-t)/(k11*tau_0))); # <- this is for increasing load, be careful


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
	

# (((1+r*(k^2))/(1+r))^x)*(1-e^((-t)/(k11*tau_0)))
	
# Failures provided by GP
# gp_failure{t in failures, k in components} : x[k,t-1] = 1;
