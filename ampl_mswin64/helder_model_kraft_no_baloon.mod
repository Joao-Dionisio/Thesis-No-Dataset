param Tmax;                     # upper bound on the number of periods
set components = {"OPS", "Cooling_System", "Oil", "Winding"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price (of electricity?)
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param load_effect;                              # equipment load usage factor
param periodicity;					  # periodicity of preventive maintenance


#param squared_load {0..20} := 0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0, 1.21, 1.44, 1.69, 1.96, 2.25, 2.56, 2.89, 3.24, 3.61, 4.0;

param possible_load{0..20};
param squared_load{0..20};

# Trying to linearize the exponent
#param V1 >= 0; # this is the slope
#param V2 >= V1;
#param V3 >= V2;
#param V4 >= V3;
#param V5 >= V4;
#param V6 >= V5;
#param V7 >= V6;
#param V8 >= V7;
#param V9 >= V8;
#param V10 >= V9;
#param V11 >= V10;


#param hs_t1 >= 0; # this is the hotspot temperature
#param hs_t2 >= hs_t1;
#param hs_t3 >= hs_t2;
#param hs_t4 >= hs_t3;
#param hs_t5 >= hs_t4;
#param hs_t6 >= hs_t5;
#param hs_t7 >= hs_t6;
#param hs_t8 >= hs_t7;
#param hs_t9 >= hs_t8;
#param hs_t10 >= hs_t9;
#param hs_t11 >= hs_t10;

param r; # these are dependant on PT type (type of cooling system and whatnot)
param x;
param y;
param H;
param gr;
param k_11;
param k_21;
param k_22;
param tau_0;
param tau_w;

param A;
param oilsat23;
param EA;


param hs {i in 1..11};
param V {i in 1..11} = (2^((hs[i]-98)/6))/hs[i];


#param base0 >= 0;
#param base1 >= base0;
#param base2 >= base1;
#param base3 >= base2;
#param base4 >= base3;
#param base5 >= base4;
#param base6 >= base5;
#param base7 >= base6;
#param base8 >= base7;
#param base9 >= base8;
#param base10 >= base9;
#param base11 >= base10;
#param base12 >= base11;
#param base13 >= base12;
#param base14 >= base13;
#param base15 >= base14;
#param base16 >= base15;
#param base17 >= base16;
#param base18 >= base17;
#param base19 >= base18;
#param base20 >= base19;


param slope {i in 0..20} = ((1+r*squared_load[i])/(1+r))^x;

param k_to_the_power_of_y {i in 0..20} = possible_load[i]^y;


var delta_h1 {i in 0..Tmax};
var delta_h2 {i in 0..Tmax};

#param slope0;
#param slope1;
#param slope2;
#param slope3;
#param slope4;
#param slope5;
#param slope6;
#param slope7;
#param slope8;
#param slope9;
#param slope10;
#param slope11;
#param slope12;
#param slope13;
#param slope14;
#param slope15;
#param slope16;
#param slope17;
#param slope18;
#param slope19;
#param slope20;



var load_activation {0..20, 1..Tmax} binary;
#var q_squared {1..Tmax};# symbolic in squared_load; 

#var V {1..Tmax} >= 0;
var hs_temperature{0..Tmax}; # Hot spot temperature. Variable with load and used in winding degradation


var q{1..Tmax};# symbolic in squared_load;                     # quantity produced/sold in period t
var qmax{1..Tmax}>=0;                  # adjusted capacity for period t
var rul{k in components,0..Tmax} >= 0, <= RULmax[k];   # remaining useful life for item k in components
var chi{k in components,1..Tmax} binary;            # maintenance/replacement of parts k in components
var force_shutdown{1..Tmax} binary;

maximize profit:
    sum {t in 1..Tmax} (Price*q[t]-sum{k in components} Cost[k]*chi[k,t]);

# Assuming PT is brand new
RUL0{k in components}: 
    rul[k,0] = RULmax[k]; 
    #rul[k,0] = 0; # <- takes into consideration if it's worth it to buy an equipment

RUL1{t in 1..Tmax}: 
    rul["OPS",t] <=  rul["OPS",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["OPS"]*chi["OPS",t];

RUL2{t in 1..Tmax}: 
    rul["Cooling_System",t] <=  rul["Cooling_System",t-1]*d - load_effect*q[t]/Qmax + 3*RULmax["Cooling_System"]*chi["Cooling_System",t]; 

RUL3{t in 1..Tmax}: 
    rul["Oil",t] <=  rul["Oil",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["Cooling_System",t]/RULmax["Cooling_System"] - load_effect*rul["OPS",t]/RULmax["OPS"] + 4*RULmax["Oil"]*chi["Oil",t]; 

RUL4{t in 1..Tmax}:     
    #rul["Winding",t] <= rul["Winding",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["joints",t]/RULmax["joints"] - load_effect*rul["oil",t]/RULmax["oil"] + 4*RULmax["Winding"]*x["Winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced

    #rul["Winding",t] <= 1/((1/rul["Winding",t-1])+A/(exp(EA/(8.314*(hs_temperature[t]+273.15))))) + 4*RULmax["Winding"]*chi["Winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced
    #rul["Winding",t] <= rul["Winding",t-1] - 2^((hs_temperature[t]-98)/6) + 4*RULmax["Winding"]*chi["Winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced
    #rul["Winding",t] <= rul["Winding",t-1] + 4*RULmax["Winding"]*chi["Winding",t] - force_shutdown[t]*0.00001 - <<hs_t1,hs_t2,hs_t3,hs_t4,hs_t5,hs_t6,hs_t7,hs_t8,hs_t9,hs_t10; V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11>> hs_temperature[t]; # constant term has to disappear with shutdown, otherwise maintenance is forced
    rul["Winding",t] <= rul["Winding",t-1] + 4*RULmax["Winding"]*chi["Winding",t] - force_shutdown[t]*0.00001 - <<{i in 1..10} hs[i]; {i in 1..11}V[i]>> hs_temperature[t]; # constant term has to disappear with shutdown, otherwise maintenance is forced


initial_temp:
	hs_temperature[0] = 98;
	
hot_spot_temperature_increasing_load{t in 1..Tmax}:
    #hs_temperature[t] = hs_temperature[t-1] + (((1+r*(q_squared[t]^2))/(1+r))^x)*(1-exp((-t)/(k_11*tau_0))); #<- this is for increasing load, be careful
    #hs_temperature[t] = hs_temperature[t-1] + ((1+r*q[t])/(1+r))^x;#(((1+r*(q_squared[t]^2))/(1+r))^x)*(1-exp((-t)/(k_11*tau_0))); #<- this is for increasing load, be careful	
	#hs_temperature[t] = 23.9 + <<base0,base1,base2,base3,base4,base5,base6,base7,base8,base9,base10,base11,base12,base13,base14,base15,base16,base17,base18,base19;slope0,slope1,slope2,slope3,slope4,slope5,slope6,slope7,slope8,slope9,slope10,slope11,slope12,slope13,slope14,slope15,slope16,slope17,slope18,slope19,slope20>> 38.3*((1+r*(sum {i in 0..20} load_activation[i,t]*squared_load[i]))/(1+r)); #<- this is for increasing load, be careful
	#hs_temperature[t] = (RULmax["Cooling_System"]-rul["Cooling_System",t]) + 23.9 + <<base0,base5,base10,base15;slope0,slope5,slope10,slope15,slope20>> 38.3*((1+r*(sum {i in 0..20} load_activation[i,t]*squared_load[i]))/(1+r))*(1-exp((-t)/(k_11*tau_0)));#<- this is for increasing load, be careful
	hs_temperature[t] = delta_h1[t] - delta_h2[t] + (RULmax["Cooling_System"]-rul["Cooling_System",t]) + 23.9 + <<{i in 0..15 by 5} squared_load[i];{i in 0..20 by 5} slope[i]>> 38.3*((1+r*(sum {i in 0..20} load_activation[i,t]*squared_load[i]))/(1+r))*(1-exp((-t)/(k_11*tau_0)));#<- this is for increasing load, be careful
	
	
hot_spot_temperature_decreasing_load{t in 1..Tmax}:
	hs_temperature[t] = delta_h1[t] - delta_h2[t] + (RULmax["Cooling_System"]-rul["Cooling_System",t]) + 23.9 + <<{i in 0..15 by 5} squared_load[i];{i in 0..20 by 5} slope[i]>> 38.3*((1+r*(sum {i in 0..20} load_activation[i,t]*squared_load[i]))/(1+r)) - (38.3*((1+r*(sum {i in 0..20} load_activation[i,t]*squared_load[i]))/(1+r)))*(exp((-t)/(k_11*tau_0)));#<- this is for increasing load, be careful	
	

one_load_at_a_time{t in 1..Tmax}:
	sum {i in 0..20 by 5} load_activation[i,t] = 1;

# Adjusting capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];

Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


q_is_qsquared{t in 1..Tmax}:
	q[t] <= sum {i in 0..20 by 5} load_activation[i,t]*possible_load[i];


oil_implies_joints{t in 1..Tmax}: 
	chi["OPS", t] <= chi["Oil", t];

winding_implies_oil{t in 1..Tmax}: 
	chi["Winding", t] <= chi["Oil",t]; 


everything_stops{t in 1..Tmax, k in components}: 
	rul[k,t] <= force_shutdown[t]*RULmax[k];


stopping_is_definitive{t in 1..Tmax-1}: 
	force_shutdown[t+1] <= force_shutdown[t];
	
	
# Failures provided by GP
# gp_failure{t in failures, k in components} : x[k,t-1] = 1;




# 23.9 + <<base0,base5,base10,base15;slope0,slope5,slope10,slope15,slope20>> 38.3*((1+r*(sum {i in 0..20} load_activation[i,t]*squared_load[i]))/(1+r))*(1-exp((-t)/(k_11*tau_0)));


#delta h = delta h1 + delta h2

# delta h1[t] = delta h1[t-1] + (k21*Hgr*<<base0,base5,base10,base15,base20;>>K^y - delta h1[t-1])*(1-e^((-t)/(k22*tau_w)))
# delta h2[t] = delta h2[t-1] + ((k21-1)*Hgr*K^y - delta h2[t-1])*(1-e^((-t)/(tau0/k22)))


# delta h1[t] = k21*Hgr*K^y + (delta h1[t-1] - k21*Hgr*K^y)*e^((-t)/(k22*tau_w))

# delta h2[t] = (k21-1)*Hgr*K^y + (delta h2[t-1] - (k21-1)*Hgr*K^y)*e^((-t)/(tau0/k22))



initial_delta_h1:
	delta_h1[0] = 0;
	
initial_delta_h2:
	delta_h2[0] = 0;



delta_h1_increasing_load {t in 1..Tmax}:
	delta_h1[t] = delta_h1[t-1] + (k_21*H*gr*(sum{i in 0..20 by 5} load_activation[i,t]*possible_load[i]^y) - delta_h1[t-1])*(1-exp((-t)/(k_22*tau_w)));

delta_h2_increasing_load {t in 1..Tmax}:
	delta_h2[t] = delta_h2[t-1] + ((k_21-1)*H*gr*(sum{i in 0..20 by 5} load_activation[i,t]*possible_load[i]^y) - delta_h2[t-1])*(1-exp((-t)/(tau_0/k_22)));


delta_h1_decreasing_load {t in 1..Tmax}:
	delta_h1[t] = k_21*H*gr*(sum{i in 0..20 by 5} load_activation[i,t]*possible_load[i]^y) + (delta_h1[t-1] - k_21*H*gr*(sum{i in 0..20 by 5} load_activation[i,t]*possible_load[i]^y))*exp((-t)/(k_22*tau_w));

delta_h2_decreasing_load {t in 1..Tmax}:
	delta_h2[t] = ((k_21-1)*H*gr*(sum{i in 0..20 by 5} load_activation[i,t]*possible_load[i]^y)) + (delta_h2[t-1] - (k_21-1)*H*gr*(sum{i in 0..20 by 5} load_activation[i,t]*possible_load[i]^y))*exp((-t)/(tau_0/k_22));



# Oil change decreases temperature
#oil_change_reduce_temp {t in 1..Tmax-1}:
#	hs[t+1] <= (1-chi['Oil',t])*999;


# Wc = moist*2.24*e^(-0.04*Tsample)

## moist -> valor de H2O em ppm


# Oil change decreases temperature
#oil_change_reduce_humidity {t in 1..Tmax-1}:
#	moisture[t+1] <= (1-chi['Oil',t])*999;




# if q[t] >= q[t-1] then d else e

# q[t] >= q[t-1] + 0.001 - M*(1-indicator)
# q[t] <= q[t-1] + M*indicator
#
#
# increasing_load_temperature - M*(1-indicator) <= temp_temperature[t] <= increasing_load_temperature + M*(1-indicator)
# decreasing_load_temperature - M*indicator <= temp_temperature[t] <= decreasing_load_temperature + M*indicator
#
#


data;
param:   RULmax   Cost :=
   OPS	  30   1900000
   Cooling_System 15 122000
   Oil  	  15   60000
   Winding 	  50   100000 # values from markovian model (in pounds)
;				# the winding RUL is being used as the DP RUL


# por cada 6 graus de aumento, o V duplica
#param: hot_spot_temp paper_insulation :=
#	80	0.125
#	86	0.25
#	92	0.5
#	98	1
#	104	2
#	110	4
#	116	8
#	122	16
#	128	32
#	134	64
#	140	128


#set numbers_1_to_20 := 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;

param: possible_load squared_load :=
0	0	0
1	0.1	0.01
2	0.2	0.04
3	0.3	0.09
4   0.4	0.16
5   0.5	0.25
6	0.6	0.36
7   0.7	0.49
8   0.8	0.64
9   0.9	0.81
10  1	1.0
11  1.1	1.21
12  1.2	1.44
13  1.3	1.69
14  1.4	1.96
15  1.5	2.25
16  1.6	2.56
17  1.7	2.89
18  1.8	3.24
19  1.9	3.61
20  2	4.0;



#param V1 := 0.02083; # this is the slope
#param V2 := 0.04166;
#param V3 := 0.08333;
#param V4 := 0.16666;
#param V5 := 0.33333;
#param V6 := 0.66666;
#param V7 := 1.33333;
#param V8 := 2.66666;
#param V9 := 5.33333;
#param V10 := 10.6666;
#param V11 := 21.3333;

#param hs_t1 := 80; # this is the hotspot temperature
#param hs_t2 := 86;
#param hs_t3 := 92;
#param hs_t4 := 98;
#param hs_t5 := 104;
#param hs_t6 := 110;
#param hs_t7 := 116;
#param hs_t8 := 122;
#param hs_t9 := 128;
#param hs_t10 := 134;
#param hs_t11 := 140;

param hs :=
1	80
2	86
3	92
4	98
5	104
6	110
7	116
8	122
9	128
10	134
11	140;



#param base0 := 0.000999;
#param base1 := 0.001;
#param base2 := 0.0026;
#param base3 := 0.009;
#param base4 := 0.0266;
#param base5 := 0.0634;
#param base6 := 0.130;
#param base7 := 0.241;
#param base8 := 0.410;
#param base9 := 0.656;
#param base10 := 1;
#param base11 := 1.464;
#param base12 := 2.07;
#param base13 := 2.85;
#param base14 := 3.84;
#param base15 := 5.06;
#param base16 := 6.55;
#param base17 := 8.34;
#param base18 := 10.5;
#param base19 := 13;
#param base20 := 16;


#param slope0 := 3.98;
#param slope1 := 3.91;
#param slope2 := 3.29;
#param slope3 := 2.56;
#param slope4 := 2.07;
#param slope5 := 1.74;
#param slope6 := 1.50;
#param slope7 := 1.33;
#param slope8 := 1.20;
#param slope9 := 1.09;
#param slope10 := 1;
#param slope11 := 0.927;
#param slope12 := 0.846;
#param slope13 := 0.810;
#param slope14 := 0.764;
#param slope15 := 0.723;
#param slope16 := 0.687;
#param slope17 := 0.645;
#param slope18 := 0.625;
#param slope19 := 0.599;
#param slope20 := 0.574;

param Tmax := 100;
param Price := 1000;
param Qmax := 2000000;
param d := 0.9;
param load_effect := 1;

# from IEEE standard
param r = 1000; # these are dependant on PT type (type of cooling system and whatnot)
param x = 0.8;
param y = 1.3;
param k_11 = 0.5;
param k_21 = 2.0;
param k_22 = 2.0;
param tau_0 = 210;
param tau_w = 10;

param H = 1.4;
param gr = 14.5;
#param t_w = 7;



# from helder's email
param A := 460000;
param oilsat23 = 55; 
param EA = 82000;
