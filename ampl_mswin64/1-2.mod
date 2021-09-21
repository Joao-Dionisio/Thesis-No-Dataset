# Adding variable costs # try to add degradation based on age (can't multiply because of the solvers)

param c_1 := 5;


var chi {1..100} binary;

var HI {1..100};


# Assuming 100 years of lifetime
minimize cost: sum {t in 1..100}(chi[1,t]*c_1);


subject to

initial_health: HI[1] = 100;
health_threshold {t in 1..100}: 30 <= HI[t] <= 100;


# Assuming full repair and one maintenance
maintenance {t in 1..99}: HI[t+1] <= 100*chi[t] + 0.9*HI[t];