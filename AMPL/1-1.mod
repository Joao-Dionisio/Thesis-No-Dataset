var chi {0..1, 1..100} binary;

var HI {1..100};

# Assuming 100 years of lifetime
minimize cost: sum {t in 1..100}(chi[0,t]*0 + chi[1,t]*5);


subject to

initial_health: HI[1] = 100;
health_threshold {t in 1..100}: HI[t] >= 30;

one_decision {t in 1..100}: sum {i in 0..1} chi[i,t] == 1; # Ensuring one decision at every point in time


# Assuming full repair and one maintenance
small_maintenance1 {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = 100;
small_maintenance2 {t in 1..99}: chi[1,t] = 0 ==> HI[t+1] = HI[t]*0.9;
