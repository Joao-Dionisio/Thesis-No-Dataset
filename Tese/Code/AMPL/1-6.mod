## Introducing depreciation based on load

param c_0; 
param c_1;
param c_2;
param c_3;

param cost {0..3};
param cost[0] = 0;

param interest_rate;

param demand {1..100};       # index is time
param max_capacity {1..100}; # index is Health Index

var chi {0..3, 1..100} binary;
var HI {1..100};
var load {1..100};

# Assuming 100 years of lifetime
#minimize cost: sum {t in 1..100}((chi[0,t]*c_0 + chi[1,t]*c_1 + chi[2,t]*c_2 + chi[3,t]*c_3)/((1+interest_rate)^(t)));
minimize cost: sum {t in 1..100}(sum {t in 0..3}(chi[i,t]*cost[i])/((1+interest_rate)^(t)));

subject to
initial_health: HI[1] = 100;
health {t in 1..100}: 30 <= HI[t] <= 100; # failure_rate < K?

one_decision: product {t in 1..100} sum {i in 0..1} chi[i,t] == 1; # Ensuring one decision at every point in time

degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = load*HI[t]; #have degradation depend on age of transformer
aging {t in 1..99}: chi[3,t] = 0 ==> age[t+1] = age[t] + 1;

small_maintenance1 {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = HI[t]+20;
big_maintenance1 {t in 1..99}: chi[2,t] = 1 ==> HI[t+1] = HI[t] + 70;
replacement {t in 1..99}: chi[3,t] = 1 ==> HI[t+1] = 100; 

reset_age {1..99}: chi[3,t] = 1 ==> age[t+1] = 0;

energy_requirement {t in 0..100}: load[t] >= demand[t];
energy_capacity {t in 0..100}:    load[t] <= max_capacity[HI[t]];