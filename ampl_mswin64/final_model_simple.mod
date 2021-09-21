param Tmax;                     # upper bound on the number of periods
set components = {"OPS", "Cooling_System", "Oil", "Winding"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price (of electricity?)
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param load_effect;                              # equipment load usage factor
param periodicity;					  # periodicity of preventive maintenance

param interest_rate;  # For calculating PV of maintenance



param possible_load{0..20};	# := {0,0.1,0.2,...,2}. 1 is rated load
param squared_load{0..20};	# := {0,0.01,0.04,...,4}
param possible_temperature{0..20};	# PT temperature operating at a certain load



# these are dependant on PT type (type of cooling system and whatnot)
#param r; 
#param x;
#param y;
#param H;
#param gr;
#param k_11;
#param k_21;
#param k_22;
#param tau_0;
#param tau_w;

#param A;
#param oilsat23;
#param EA;


param hs {i in 1..11};
param V {i in 1..11} = (2^((hs[i]-98)/6));


# param moisture {1..3};
# param V {i in 1..11, j in 1..3} = moisture[i]*(2^((hs[i]-98)/6))/hs[i];
# <<{j in 1..3} moisture[i];<<{i in 1..10} hs[i]; {i in 1..11}V[i,j]>> real_temp[t]>>

# two piecewise linear functions!! 

param slope {i in 0..20} = ((1+r*squared_load[i])/(1+r))^x;

param k_to_the_power_of_y {i in 0..20} = possible_load[i]^y;


var delta_h1 {i in 0..Tmax};
var delta_h2 {i in 0..Tmax};



var load_activation {0..20, 1..Tmax} binary;  # Used for squared load


var proposed_temperature{0..Tmax}; # Hot spot temperature. Variable with load and used in winding degradation
var real_temp{0..Tmax}; # <- because of force_shutdown, otherwise maintenance would be mandatory

var q{1..Tmax};				 				           # quantity produced/sold in period t
var qmax{1..Tmax}>=0;                  				   # adjusted capacity for period t
var rul{k in components,0..Tmax} >= 0, <= RULmax[k];   # remaining useful life for item k in components
var chi{k in components,1..Tmax} binary;               # maintenance/replacement of parts k in components
var force_shutdown{1..Tmax} binary;

maximize profit:
    sum {t in 1..Tmax} (Price*q[t]-sum{k in components} Cost[k]*chi[k,t])/(interest_rate^t);

# Assuming PT is brand new
RUL0{k in components}: 
    rul[k,0] = RULmax[k]; 

RUL1{t in 1..Tmax}: 
    rul["OPS",t] <=  rul["OPS",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["OPS"]*chi["OPS",t];

RUL2{t in 1..Tmax}: 
    rul["Cooling_System",t] <=  rul["Cooling_System",t-1]*d - load_effect*q[t]/Qmax + 3*RULmax["Cooling_System"]*chi["Cooling_System",t]; 

RUL3{t in 1..Tmax}: 
    #rul["Oil",t] <=  rul["Oil",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["Cooling_System",t]/RULmax["Cooling_System"] - load_effect*rul["OPS",t]/RULmax["OPS"] + 4*RULmax["Oil"]*chi["Oil",t] - real_temp[t]/possible_temperature[20]; 
	rul["Oil",t] <=  rul["Oil",t-1]*d - load_effect*rul["OPS",t]/RULmax["OPS"] - real_temp[t]/possible_temperature[20] + 4*RULmax["Oil"]*chi["Oil",t];# removing load dependancy (implicit in temperature)

RUL4{t in 1..Tmax}:     
	rul["Winding",t] <= rul["Winding",t-1] + 5*RULmax["Winding"]*chi["Winding",t] - <<{i in 1..10} hs[i]; {i in 1..11} V[i]>> real_temp[t] + rul["Cooling_System",t]/RULmax["Cooling_System"];# - rul["Oil",t]/RULmax["Oil"]; 


# real_temp = proposed_temperature*force_shutdown
0implies0{t in 1..Tmax}:
	real_temp[t] <= possible_temperature[20]*force_shutdown[t];
	
ubound{t in 1..Tmax}:
	real_temp[t] <= proposed_temperature[t];
	
lbound{t in 1..Tmax}:
	real_temp[t] >= proposed_temperature[t] - (1-force_shutdown[t])*possible_temperature[20];


initial_temp:
	proposed_temperature[0] = 98;
	

hot_spot_temperature{t in 1..Tmax}:
	proposed_temperature[t] = sum{i in 0..20 by 5} load_activation[i,t]*possible_temperature[i];
	

one_load_at_a_time{t in 1..Tmax}:
	sum {i in 0..20 by 5} load_activation[i,t] = 1;

# Adjusting capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];

Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


q_is_qsquared{t in 1..Tmax}:
	q[t] = sum {i in 0..20 by 5} load_activation[i,t]*possible_load[i];


OPS_implies_oil{t in 1..Tmax}: 
	chi["OPS", t] <= chi["Oil", t];

winding_implies_oil{t in 1..Tmax}: 
	chi["Winding", t] <= chi["Oil",t]; 


everything_stops{t in 1..Tmax, k in components}: 
	rul[k,t] <= force_shutdown[t]*RULmax[k];


stopping_is_definitive{t in 1..Tmax-1}: 
	force_shutdown[t+1] <= force_shutdown[t];
	
		
#Ends when winding reaches 0
Early_stop{t in 1..Tmax}: 
	sum{i in t..Tmax} (qmax[i] + sum{k in components} chi[k,i]) <= rul["Winding",t-1]*(Qmax+3)*Tmax; 


#Can't operate with 0 health on any component
#ensure_positive_rul {t in 1..Tmax, k in components}:
#	rul["Winding",t]/RULmax[k] <= rul[k,t]; # ensures all components must be strictly positive, until we let the PT fail


# Oil change decreases temperature
#oil_change_reduce_temp {t in 1..Tmax-1}:
#	hs[t+1] <= (1-chi['Oil',t])*999;
