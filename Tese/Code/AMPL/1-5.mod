## Adding interest rate

param c_0; 
param c_1;
param c_2;
param c_3;

param interest_rate;

var chi {0..3, 1..100} binary;

var HI {1..100};

var age {1..100};

# Assuming 100 years of lifetime
minimize cost: sum {t in 1..100}((chi[0,t]*c_0 + chi[1,t]*c_1 + chi[2,t]*c_2)/((1+interest_rate)^(t)));


subject to
initial_health: HI[1] = 100;
health {t in 1..100}: 30 <= HI[t] <= 100; # failure_rate < K?

initial_age: HI[1] = 100;

one_decision: product {t in 1..100} sum {i in 0..1} chi[i,t] == 1; # Ensuring one decision at every point in time

degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = 0.9*HI[t]; #have degradation depend on age of transformer
aging {t in 1..99}: chi[3,t] = 0 ==> age[t+1] = age[t] + 1;

small_maintenance1 {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = HI[t]+20;
big_maintenance1 {t in 1..99}: chi[2,t] = 1 ==> HI[t+1] = HI[t] + 70;
replacement {t in 1..99}: chi[3,t] = 1 ==> HI[t+1] = 100; # add an age variable that resets to 0

reset_age {t in 1..99}: chi[3,t] = 1 ==> age[t+1] = 0;