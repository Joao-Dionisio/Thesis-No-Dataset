param t0;
param tfinal;

param time {t0..tfinal};

param cost;

param c_0 {0,1,2,3}; 
param c_1 {0,1,2,3};
param c_2 {0,1,2,3};
param c_3 {0,1,2,3};

param Max_capacity;
param Demand;

param m;
param M;

param interest_rate;

var chi {0..3, t0..tfinal} binary

var delta_1;
var delta_2;

var HI;
var t;
var K;
var load;

minimize z: sum {t in t0..tfinal}(sum {i in 0..3} chi[i][t]*cost[i]/((1+interest_rate)^(t - t0));


subject to
health: HI(t) >= K >0; # failure_rate < K?

one_decision: product {t in t_0..t_final} sum {i in 0..3} chi[i][t] == 1 # Ensuring one decision at every point in time

small_improv {t in 1..99}:   chi[1,t] = 1 ==> HI[t + 1] = m(HI(t));
small_downtime {t in 1..99}: chi[1,t] = 1 ==> R[t'] = 0, forall t' in t_0..t_0+delta_1;


big_improv:   chi_2 = 1 ==> HI(t + delta_2) = M(HI(t));
big_downtime: chi_2 = 1 ==> R(t') = 0, forall t' in t_0..t_0+delta_2;


energy_requirement {t in t_0..t_final}: load[t] >= demand[t];
energy_capacity {t in t_0..t_final}:    load[t] <= max_capacity[HI[t]];