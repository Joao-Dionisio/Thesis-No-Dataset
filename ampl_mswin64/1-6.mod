## Introducing depreciation based on load

param price {1..3}; # light maintenance/heavy maintenance/replacement
#let price[1] := 3;
#let price[2] := 6;
#let price[3] := 13;

param interest_rate;# := 0.1;

param small_maintenance;# := 15;
param big_maintenance;# := 70;
param degradation_factor;# := 0.9;

param t_final;# := 100;

param min_health;# := 30;
param max_health;# := 100;

param demand {1..t_final};       # index is time
param max_capacity {1..100}; # index is Health Index

set failure_time;# := {2}; # amplpy.py

var chi {1..3, 1..t_final} binary;
var HI {1..t_final};
var pt_load {1..t_final};

var age {1..t_final} >= 0 integer;

var Z {1..t_final} >= 0;
var Y {1..t_final} >= 0;
var X {1..t_final} >= 0;
var W {1..t_final} >= 0;

# Assuming t_final years of lifetime
#minimize cost: sum {t in 1..t_final}((chi[0,t]*c_0 + chi[1,t]*c_1 + chi[2,t]*c_2 + chi[3,t]*c_3)/((1+interest_rate)^(t)));
minimize cost: sum {t in 1..t_final}(sum {i in 1..3}(chi[i,t]*price[i])/((1+interest_rate)^(t-1)));

subject to
initial_health: HI[1] = max_health;

health {t in 1..t_final}: min_health <= HI[t] <= max_health; 

#reset_age {1..t_final-1}: chi[3,t] = 1 ==> age[t+1] = 0;
initial_age {t in 1..t_final}: age[1] = 1;
aging {t in 1..t_final-1}: age[t+1] = age[t] - W[t] + 1; # no replacement => aging; no replacement => age 1

pt_failure {t in failure_time}: HI[t] = min_health;
#health_evolution {t in 1..t_final-1}: HI[t+1] = degradation_factor*HI[t] - degradation_factor*Z[t] + Z[t] + small_maintenance*chi[1,t] + Y[t] + big_maintenance*chi[2,t] + max_health*chi[3,t] - age[t] - pt_load[t]; # no maintenance => degradation; maintenance => all ok
health_evolution {t in 1..t_final-1}: HI[t+1] = degradation_factor*HI[t] + Z[t] + small_maintenance*chi[1,t] - degradation_factor*Z[t]  + Y[t] + big_maintenance*chi[2,t] - degradation_factor*Y[t] + max_health*chi[3,t] - degradation_factor*chi[3,t] - X[t] + degradation_factor*X[t] - age[t] - pt_load[t]; # no maintenance => degradation; maintenance => all ok

energy_requirement {t in 1..t_final}: pt_load[t] >= t/3; #demand[t];
energy_capacity {t in 1..t_final}:    pt_load[t] <= HI[t]*2; #max_capacity[HI[t]]; # <- this is what we want, but variable in index is impossible
  

## W is chi[3,t]*age <- replacement
0implies0age {t in 1..t_final-1}: W[t] <= t_final*chi[3,t]; # Assuming max age of t_final
uboundage {t in 1..t_final-1}: W[t] <= age[t]; # W <= age, regardless of chi
lboundage {t in 1..t_final-1}: W[t] >= age[t] - (1-chi[3,t])*t_final; # chi is 1, W = age
 
## Z is chi[1,t]*HI[t] <- small maintenance
0implies01 {t in 1..t_final-1}: Z[t] <= max_health*chi[1,t]; # If chi = 0, Z = 0
ubound1 {t in 1..t_final-1}: Z[t] <= HI[t]; # Z <= HI, regardless of chi
lbound1 {t in 1..t_final-1}: Z[t] >= HI[t] - (1-chi[1,t])*max_health; # chi is 1, Z = HI
 

## Y is chi[2,t]*HI[t] <- large maintenance
0implies02 {t in 1..t_final-1}: Y[t] <= max_health*chi[2,t]; # If chi = 0, Y = 0
ubound2 {t in 1..t_final-1}: Y[t] <= HI[t]; # Y <= HI, regardless of chi
lbound2 {t in 1..t_final-1}: Y[t] >= HI[t] - (1-chi[2,t])*max_health; # chi is 1, Y = HI
 

## X is Z[t]*chi[2,t] <- for inclusion exclusion principle
0implies0chi {t in 1..t_final-1}: chi[2,t] <= chi[2,t]*max_health;
uboundchi {t in 1..t_final-1}: X[t] <= Z[t];
lboundchi{t in 1..t_final-1}: X[t] >= Z[t] - (1-chi[2,t])*max_health;



################################
#Explanation of health evolution
#
# chi[1,t] | chi[2,t] | chi[3,t] | HI[t+1]
#-----------------------------------------
#	0	   |  	0	  |		0	 | degradation_factor*HI[t]
#------------------------------------------
#	1	   |  	0	  |		0	 | HI[t] + small_maintenance
#------------------------------------------
#	0	   |  	1	  |		0	 | HI[t] + large_maintenance
#------------------------------------------
#	1	   |  	1	  |		0	 | HI[t] + small_maintenance + large_maintenance
#------------------------------------------
#	0	   |  	0	  |		1	 | max_health
#------------------------------------------


# ^ no other combination makes sense

# With no maintenance: HI[t+1] = HI[t]*degradation_factor
# With maintenance of type 1: HI[t+1] = HI[t]*degradation_factor - chi[1,t]*HI[t]*degradation_factor + chi[1,t]*(HI[t]+small_maintenance)
# With maintenance of type 2: HI[t+1] = HI[t]*degradation_factor - chi[2,t]*HI[t]*degradation_factor + chi[2,t]*(HI[t]+large_maintenance)

# For maintenance of type 1 and 2 we have to be careful, as we have overlapping events and so we need to subtract their intersection
# With maintenance of type 1 and 2: HI[t+1] = HI[t]*degradation_factor - chi[1,t]*HI[t]*degradation_factor + chi[1,t]*(HI[t]+small_maintenance) - chi[2,t]*HI[t]*degradation_factor + chi[2,t]*(HI[t]+large_maintenance) - chi[1,t]*chi[2,t]*HI[t] + chi[1,t]*chi[2,t]*HI[t]*degradation_factor

# With everything:  HI[t+1] = HI[t]*degradation_factor - chi[1,t]*HI[t]*degradation_factor + chi[1,t]*(HI[t]+small_maintenance) - chi[2,t]*HI[t]*degradation_factor + chi[2,t]*(HI[t]+large_maintenance) - chi[1,t]*chi[2,t]*HI[t] + chi[1,t]*chi[2,t]*HI[t]*degradation_factor - chi[3,t]*HI[t]*degradation_factor + chi[3,t]*max_health

# Then I created variables W,X,Y,Z to represent each multiplication of variables that appear

# With substitution: HI[t+1] = degradation_factor*HI[t] + Z[t] + small_maintenance*chi[1,t] - degradation_factor*Z[t]  + Y[t] + big_maintenance*chi[2,t] - degradation_factor*Y[t] - X[t] + degradation_factor*X[t] + max_health*chi[3,t] - degradation_factor*chi[3,t]


option solver gurobi;
#model 1-6.mod;
data scenario1.dat;
solve;
reset;

