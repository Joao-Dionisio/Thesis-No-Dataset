# Adding variable costs # try to add degradation based on age (can't multiply because of the solvers)

param c_0; 
param c_1;


var chi {0..1, 1..100} binary;

var HI {1..100};


# Assuming 100 years of lifetime
minimize cost: sum {t in 1..100}(chi[0,t]*c_0 + chi[1,t]*c_1);


subject to

initial_health: HI[1] = 100;
health_threshold {t in 1..100}: 30 <= HI[t] <= 100;

one_decision {t in 1..100}: sum {i in 0..1} chi[i,t] == 1; # Ensuring one decision at every point in time


degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = 0.9*HI[t];

# Assuming full repair and one maintenance
condition {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = 100;
