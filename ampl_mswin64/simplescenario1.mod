#param c_0 {0,1,2,3}; 
#param c_1 {0,1,2,3};
#param c_2 {0,1,2,3};
#param c_3 {0,1,2,3};

#param Max_capacity;
#param Demand;

#param m;
#param M;

#param interest_rate;

var chi {0..1, 1..100} binary;

#param HI {1..100};
var HI {1..100};

let HI[1] := 100;
#let HI[1] := 100;

#let {t in 1..99} HI[t+1] := 0.9*HI[t];
#let {t in 1..99} HI[t+1] := if chi[1,t] = 0 then 0.9*HI[t] else 100;

#var t;
#var load;
# Assuming 100 years of lifetime
minimize cost: sum {t in 1..100}(chi[0,t]*0 + chi[1,t]*5);


subject to
#health {t in 1..100}: HI[t] >= 30; # failure_rate < K?
#health {t in 1..100}: HI[t] <= 30 ==> chi[1,t] >= 1 else chi[1,t] <= 0; # failure_rate < K?
#maint {t in 1..100}: chi[1,t] = if HI[t] >= 30 then 1 else 0;
initial_health: HI[1] = 100;
#health_threshold {t in 1..100}: sum{i in 0..1} HI[t]*chi[i,t] >= 31; 
#health_threshold {t in 1..100}: HI[t] + chi[1,t] >= 31; 
health_threshold {t in 1..100}: 30 <= HI[t] <= 100;

one_decision {t in 1..100}: sum {i in 0..1} chi[i,t] == 1; # Ensuring one decision at every point in time

degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = 0.9*HI[t];
# Assuming full repair and one maintenance
#small_improv {t in 1..100}:   chi[1,t] = 1	 ==> HI[t] := 100;
#small_downtime: chi_1 = 1 ==> R(t') = 0, forall t' in t_0..t_0+delta_1;
#condition {t in 1..100}: HI[t+1] := if chi[1,t] = 0 then 0.9*HI[t] else 100;
condition {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = 100;
#condition2 {t in 1..99}: chi[1,t] = 0 ==> HI[t+1] = HI[t]*0.9;

#energy_requirement {t in 0..100}: load[t] >= 50; # assuming constant demand
#energy_capacity {t in 0..100}:    load[t] <= max_capacity[HI];