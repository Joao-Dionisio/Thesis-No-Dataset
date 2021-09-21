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

var chi_0 {t0..tfinal} binary; # Change this so it's an array
var chi_1 {t0..tfinal} binary;
var chi_2 {t0..tfinal} binary;
var chi_3 {t0..tfinal} binary;

#var chi {1..4, t0..tfinal} binary

var delta_1;
var delta_2;

var HI;
var t;
var K;
var load;

#fuction chi(i)  { return if i == 0 then chi_0 else if i == 1 then chi_1 else if i == 2 then chi_2 else chi_3;}
#fuction cost(i) { return if i == 0 then cost_0 else if i == 1 then cost_1 else if i == 2 then cost_2 else cost_3;}


#minimize z: sum {t in t0..tfinal}(sum {i in 0..4} chi(i)*cost(i));

minimize z: sum {t in t0..tfinal}(sum {i in 0..3} chi[i][t]*cost[i]/((1+interest_rate)^(t - t0));


subject to
health: HI(t) >= K >0; # failure_rate < K?

one_decision: product {t in t_0..t_final} sum {i in 0..3} chi[i][t] == 1 # Ensuring one decision at every point in time

small_improv:   chi_1 = 1 ==> HI(t + delta_1) = m(HI(t));
small_downtime: chi_1 = 1 ==> R(t') = 0, forall t' in t_0..t_0+delta_1;


big_improv:   chi_2 = 1 ==> HI(t + delta_2) = M(HI(t));
big_downtime: chi_2 = 1 ==> R(t') = 0, forall t' in t_0..t_0+delta_2;


energy_requirement {t in t_0..t_final}: load[t] >= demand[t];
energy_capacity {t in t_0..t_final}:    load[t] <= max_capacity[HI[t]];