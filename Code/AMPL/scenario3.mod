param cost;
param revenue {1..100};

#param c_0; 
#param c_1;
#param c_2;
#param c_3;
param cost {0..3};

param max_capacity {1..100}; # Index is Health
param demand {1..100}; # Index is time

var chi {0..3, 1..100} binary;

var delta_1;
var delta_2;

var HI;
var t;
var K;
var load;
var profit;
var age {1..100}; # <- probably param age



maximize profit: sum {1..100} revenue[t] - cost[t];

subject to

health_cap {t in 1..100}: HI[t] <= 100;

one_decision {t in 1..100}: sum {i in 0..3} chi[i][t] == 1 # Ensuring one decision at every point in time


degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = HI[t] - load[t] - age[t]; # if no maintenance, it degrades
aging {t in 1..99}:       chi[3,t] = 0 ==> age[t+1] = age[t] + 1; # if no replacement, it ages
reset_age {t in 1..99}: chi[3,t] = 1 ==> age[t+1] = 0;


## Maintenance and Replacement
small_improv {t in 1..99}:   chi[1,t] = 1 ==> HI[t+1] = HI[t] - 10;
small_downtime {t in 1..99}: chi[1,t] = 1 ==> R[t] = some formula - small thing;

big_improv {1..99}:   chi[2,t] = 1 ==> HI[t + 1] = HI[t] - 30;
big_downtime {t in 1..99}: chi[2,t] = 1 ==> R[t] = some formula - big thing;

replacement {t in 1..99}: chi[3,t] = 1 ==> HI[t+1] = 100; 
replacement_downtime {t in 1..99}: chi[3,t] = 1 ==> R[t] = some formula - very big thing;


## Load Constraints
energy_requirement {t in 0..100}: load[t] >= demand[t];
energy_capacity {t in 0..100}:    load[t] <= max_cap[HI[t]];

## Formulas to facilitate
#cost_formula {1..100}: sum {t in 1..100}((chi[0,t]*c_0 + chi[1,t]*c_1 + chi[2,t]*c_2 + chi[3,t]*c_3)/((1+interest_rate)^(t)));
cost_formula {1..100}:  sum {t in 1..100} (sum {i in 0..3} chi[i,t]*cost[i])/((1+interest_rate)^(t));
revenue_formula {1..100}: chi[0,t] = 1 ==> revenue[t] = 10 + load[t]; # random formula