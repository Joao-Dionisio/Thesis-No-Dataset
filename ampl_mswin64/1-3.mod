# Adding heavy maintenance and interest rate

#param c_1 := 5;
#param c_2 := 7;
param price {1..2};
let price[1] := 5;
let price[2] := 7;

param interest_rate := 0;

var chi {1..2, 1..100} binary;

var HI {1..100};
var Z {1..100};
var Y {1..100};

# Assuming 100 years of lifetime
#minimize cost: sum {t in 1..100}((chi[1,t]*c_1 + chi[2,t]*c_2)/((1+interest_rate)^(t-1)));
minimize cost: sum {t in 1..100} sum {i in 1..2}(chi[i,t]*price[i])/((1+interest_rate)^(t-1));

subject to
initial_health: HI[1] = 100;
health_threshold {t in 1..100}: 30 <= HI[t] <= 100;

#one_decision {t in 1..100}: sum {i in 1..2} chi[i,t] == 1; # Ensuring one decision at every point in time

#degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = 0.9*HI[t];

# Assuming full repair and one maintenance
#small_maintenance {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = HI[t]+20;
#big_maintenance {t in 1..99}: chi[2,t] = 1 ==> HI[t+1] = HI[t]+70; # have to set HI to a maximum of 100


#small_maintenance {t in 1..99}: HI[t] <= 40*chi[1,t] + HI[t]*0.9;
#big_maintenance {t in 1..99}: HI[t] <= 70*chi[2,t] + HI[t]*0.9;

#small_maintenance1 {t in 1..99}: HI[t+1] <= 100*chi[1,t];
#small_maintenance2 {t in 1..99}: HI[t+1] >= 0.9*HI + (1-chi[1,t]*70)
#small_maintenance2 {t in 1..99}: HI[t+1] >= 0.9*HI + 


## Z is chi[1,t]*HI[t]
 0implies01 {t in 1..99}: Z[t] <= 100*chi[1,t]; # If chi = 0, Z = 0
 ubound1 {t in 1..99}: Z[t] <= HI[t]; # Z <= HI, regardless of chi
 lbound1 {t in 1..99}: Z[t] >= HI[t] - (1-chi[1,t])*100; # chi is 1, Z = HI
 small_maintenance {t in 1..99}: HI[t+1] <= 0.9*HI[t] - 0.9*Z[t] + Z[t] + 15*chi[1,t] + Y[t] + 70*chi[2,t]; # no maintenance => degradation; maintenance => all ok

## Y is chi[2,t]*HI[t]
 0implies02 {t in 1..99}: Y[t] <= 100*chi[2,t]; # If chi = 0, Y = 0
 ubound2 {t in 1..99}: Y[t] <= HI[t]; # Y <= HI, regardless of chi
 lbound2 {t in 1..99}: Y[t] >= HI[t] - (1-chi[2,t])*100; # chi is 1, Y = HI
 #big_maintenance {t in 1..99}: HI[t+1] <= 0.9*HI[t] - 0.9*Y[t] + Y[t] + 70*chi[2,t]; # no maintenance => degradation; maintenance => all ok