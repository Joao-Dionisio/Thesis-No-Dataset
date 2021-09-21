## Introducing probability based failure

param price {1..3};
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

#param demand {1..t_final};       # index is time
#param max_capacity {1..100}; # index is Health Index

var chi {1..3, 1..t_final} binary;
var HI {1..t_final};
var pt_load {1..t_final};

var age {1..t_final} >= 0 integer;

var Z {1..t_final} >= 0;
var Y {1..t_final} >= 0;
var X {1..t_final} >= 0;
var W {1..t_final} >= 0;

# Assuming t_final years of lifetime
minimize cost: sum {t in 1..t_final}(sum {i in 1..3}(chi[i,t]*price[i])/((1+interest_rate)^(t-1)));

subject to
initial_health: HI[1] = 100;
health {t in 1..t_final}: min_health <= HI[t] <= max_health; # failure_rate < K?

#reset_age {1..t_final-1}: chi[3,t] = 1 ==> age[t+1] = 0;
initial_age {t in 1..t_final}: age[1] = 1;
aging {t in 1..t_final-1}: age[t+1] = age[t] - W[t] + 1; # no replacement => aging; no replacement => age 1


health_evolution {t in 1..t_final-1}: HI[t+1] = degradation_factor*HI[t] - degradation_factor*Z[t] + Z[t] + small_maintenance*chi[1,t] + Y[t] + big_maintenance*chi[2,t] + max_health*chi[3,t] - age[t] - pt_load[t]; # no maintenance => degradation; maintenance => all ok

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
 
## trying to model failure
#  