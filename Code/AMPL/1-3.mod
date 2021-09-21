# Adding heavy maintenance and interest rate

param c_0; #param costs {0..2}
param c_1;
param c_2;


param interest_rate;

var chi {0..2, 1..100} binary;

var HI {1..100};


# Assuming 100 years of lifetime
minimize cost: sum {t in 1..100}((chi[0,t]*c_0 + chi[1,t]*c_1 + chi[2,t]*c_2)/((1+interest_rate)^(t-1));


subject to
initial_health: HI[1] = 100;
health_threshold {t in 1..100}: 30 <= HI[t] <= 100;

one_decision {t in 1..100}: sum {i in 0..1} chi[i,t] == 1; # Ensuring one decision at every point in time

degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = 0.9*HI[t];

# Assuming full repair and one maintenance
small_maintenance {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = HI[t]+20;
big_maintenance {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = HI[t]+70; # have to set HI to a maximum of 100
