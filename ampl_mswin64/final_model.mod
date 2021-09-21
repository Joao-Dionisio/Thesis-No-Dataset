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



param hs {i in 0..11};
param m {1..3};
param V {0..11,0..3};


# two piecewise linear functions!! 


var x {1..Tmax} binary;
var y {1..Tmax} binary;
var z {1..Tmax} binary;

var load_activation {0..11, 1..Tmax} binary;  # Used for squared load <- you need to remove it, no longer needed


var proposed_temperature{0..Tmax}; # Hot spot temperature. Variable with load and used in winding degradation
var real_temp{0..Tmax}; # <- because of force_shutdown, otherwise maintenance would be mandatory

var q{1..Tmax};				 				           # quantity produced/sold in period t
var qmax{1..Tmax}>=0;                  				   # adjusted capacity for period t
var rul{k in components,0..Tmax} >= 0, <= RULmax[k];   # remaining useful life for item k in components
var chi{k in components,1..Tmax} binary;               # maintenance/replacement of parts k in components
var force_shutdown{1..Tmax} binary;





maximize profit:
    sum {t in 1..Tmax} (Price*q[t]-sum{k in components} Cost[k]*chi[k,t])/(interest_rate^0);

# Assuming PT is brand new
RUL0{k in components}: 
    rul[k,0] = RULmax[k]; 

#RUL1{t in 1..Tmax}: 
#    rul["Cooling_System",t] <=  rul["Cooling_System",t-1]*d - load_effect*q[t]/Qmax + 3*RULmax["Cooling_System"]*chi["Cooling_System",t]; #- load_effect*q[t]/Qmax 

#RUL2{t in 1..Tmax}: 
#    rul["OPS",t] <= rul["OPS",t-1]*d - load_effect*q[t]/Qmax + 2*RULmax["OPS"]*chi["OPS",t]; #- load_effect*q[t]/Qmax 

#RUL3{t in 1..Tmax}: 
    #rul["Oil",t] <=  rul["Oil",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["Cooling_System",t]/RULmax["Cooling_System"] - load_effect*rul["OPS",t]/RULmax["OPS"] + 4*RULmax["Oil"]*chi["Oil",t] - real_temp[t]/possible_temperature[20]; 
#	rul["Oil",t] <= rul["Oil",t-1]*d - real_temp[t]/hs[11] + 4*RULmax["Oil"]*chi["Oil",t];# removing load dependancy (implicit in temperature)

RUL41{t in 1..Tmax}:     
	rul["Winding",t] <= rul["Winding",t-1] + 500*RULmax["Winding"]*chi["Winding",t] - <<{i in 0..10} hs[i]; {i in 0..11}V[i,1]>> real_temp[t] + rul["Cooling_System",t]/RULmax["Cooling_System"] + (1-z[t])*10*RULmax["Winding"];
	
RUL42{t in 1..Tmax}:
	rul["Winding",t] <= rul["Winding",t-1] + 500*RULmax["Winding"]*chi["Winding",t] - <<{i in 1..10} hs[i]; {i in 1..11}V[i,2]>> real_temp[t] + rul["Cooling_System",t]/RULmax["Cooling_System"] + (1-y[t])*10*RULmax["Winding"];# - rul["Oil",t]/RULmax["Oil"];
	
RUL43{t in 1..Tmax}:
	rul["Winding",t] <= rul["Winding",t-1] + 500*RULmax["Winding"]*chi["Winding",t] - <<{i in 1..10} hs[i]; {i in 1..11}V[i,3]>> real_temp[t] + rul["Cooling_System",t]/RULmax["Cooling_System"] + (1-x[t])*10*RULmax["Winding"];
	



#force_x_zero {t in 1..Tmax}:
#	rul["OPS",t]/RULmax["OPS"] <= 1/3 + 2*(1-x[t]);

	
#force_y_zero {t in 1..Tmax}:
#	rul["OPS",t]/RULmax["OPS"] <= 2/3 + 2*(1-y[t]);


#force_z_zero {t in 1..Tmax}:
#	rul["OPS",t]/RULmax["OPS"] <= 1 + 2*(1-z[t]);


#force_x_one {t in 1..Tmax}:
#	rul["OPS",t]/RULmax["OPS"] >= 1/3 - 2*x[t];
	
#force_y_one {t in 1..Tmax}:
#	rul["OPS",t]/RULmax["OPS"] >= 2/3 - 2*y[t];
	
#force_z_one {t in 1..Tmax}:
#	rul["OPS",t]/RULmax["OPS"] >= 1.1 - 2*z[t];
	
#disjoint_states {t in 1..Tmax}:
#	x[t]+y[t]+z[t] = 1; 
	
	
# real_temp = proposed_temperature*force_shutdown
0implies0{t in 1..Tmax}:
	real_temp[t] <= hs[11]*force_shutdown[t];
	
ubound{t in 1..Tmax}:
	real_temp[t] <= proposed_temperature[t];
	
lbound{t in 1..Tmax}:
	real_temp[t] >= proposed_temperature[t] - (1-force_shutdown[t])*hs[11];


initial_temp:
	proposed_temperature[0] = 98;
	

hot_spot_temperature{t in 1..Tmax}:
	proposed_temperature[t] = sum{i in 0..11} load_activation[i,t]*hs[i];
	

one_load_at_a_time{t in 1..Tmax}:
	sum {i in 0..11} load_activation[i,t] = 1;

# Adjusting capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];

Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


q_is_qsquared{t in 1..Tmax}:
	q[t] = sum {i in 0..11} load_activation[i,t]*possible_load[i];


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


maintenance_on_months{t in 1..Tmax, k in components}: 
	chi[k,t] <= 1-ceil(t mod 720/720);


#Can't operate with 0 health on any component
#ensure_positive_rul {t in 1..Tmax, k in components}:
#	rul["Winding",t]/RULmax[k] <= rul[k,t]; # ensures all components must be strictly positive, until we let the PT fail


# Oil change decreases temperature
#oil_change_reduce_temp {t in 1..Tmax-1}:
#	hs[t+1] <= (1-chi['Oil',t])*999;
