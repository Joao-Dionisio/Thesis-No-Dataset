## Adding replacement. here we do need to add the age to the degradation

#param c_0 = 0; 
#param c_1 = 5;
#param c_2 = 15;
#param c_3 = 200;

#param interest_rate = 0.05;

#var chi {0..3, 1..100} binary;

#var HI {1..100};

#var age {1..100};

# Assuming 100 years of lifetime
#minimize cost: sum {t in 1..100}(chi[0,t]*c_0 + chi[1,t]*c_1 + chi[2,t]*c_2 + chi[3,t]*c_3)/((1+interest_rate)^(t-1));


#subject to
#initial_health: HI[1] = 100;
#health {t in 1..100}: 30 <= HI[t] <= 100; 

#initial_age: age[1] = 0;

#one_decision {t in 1..100}: sum {i in 0..3} chi[i,t] == 1; # Ensuring one decision at every point in time

#degradation {t in 1..99}: chi[0,t] = 1 ==> HI[t+1] = HI[t] - age[t]; #have degradation depend on age of transformer
#aging {t in 1..99}: chi[3,t] = 0 ==> age[t+1] = age[t] + 1;

#small_maintenance1 {t in 1..99}: chi[1,t] = 1 ==> HI[t+1] = HI[t]+20;
#big_maintenance1 {t in 1..99}: chi[2,t] = 1 ==> HI[t+1] = HI[t] + 70;
#replacement {t in 1..99}: chi[3,t] = 1 ==> HI[t+1] = 100; # add an age variable that resets to 0
#reset_health {t in 1..99}: chi[3,t] = 1 ==> age[t+1] = 0;



# Adding heavy maintenance and interest rate

#param c_1 := 5;
#param c_2 := 7;
#param c_3 = 10;
param price{1..3};
let price[1] := 5;
let price[2] := 7;
let price[3] := 10;

param interest_rate := 0;

var chi {1..3, 1..100} binary;

var age {1..100} >= 0 integer;

var HI {1..100} >= 0;
var Z {1..100} >= 0;
var Y {1..100} >= 0;
var X {1..100} >= 0;
var W {1..100} >= 0;

# Assuming 100 years of lifetime
minimize cost: sum {t in 1..100}(sum {i in 1..3}(chi[i,t]*price[i])/((1+interest_rate)^(t-1)));


subject to
initial_health: HI[1] = 100;
health_threshold {t in 1..100}: 30 <= HI[t] <= 100;

initial_age {t in 1..100}: age[1] = 1;
aging {t in 1..99}: age[t+1] = age[t] - W[t] + 1; # no replacement => aging; no replacement => age 1


health_evolution {t in 1..99}: HI[t+1] <= 0.9*HI[t] - 0.9*Z[t] + Z[t] + 15*chi[1,t] + Y[t] + 70*chi[2,t] + 100*chi[3,t] - age[t]; # no maintenance => degradation; maintenance => all ok

## W is chi[3,t]*age <- replacement
0implies0age {t in 1..99}: W[t] <= 100*chi[3,t]; # Assuming max age of 100
uboundage {t in 1..99}: W[t] <= age[t]; # W <= age, regardless of chi
lboundage {t in 1..99}: W[t] >= age[t] - (1-chi[3,t])*100; # chi is 1, W = age

 
## Z is chi[1,t]*HI[t] <- small maintenance
 0implies01 {t in 1..99}: Z[t] <= 100*chi[1,t]; # If chi = 0, Z = 0
 ubound1 {t in 1..99}: Z[t] <= HI[t]; # Z <= HI, regardless of chi
 lbound1 {t in 1..99}: Z[t] >= HI[t] - (1-chi[1,t])*100; # chi is 1, Z = HI

## Y is chi[2,t]*HI[t] <- large maintenance
 0implies02 {t in 1..99}: Y[t] <= 100*chi[2,t]; # If chi = 0, Y = 0
 ubound2 {t in 1..99}: Y[t] <= HI[t]; # Y <= HI, regardless of chi
 lbound2 {t in 1..99}: Y[t] >= HI[t] - (1-chi[2,t])*100; # chi is 1, Y = HI
 #big_maintenance {t in 1..99}: HI[t+1] <= 0.9*HI[t] - 0.9*Y[t] + Y[t] + 70*chi[2,t]; # no maintenance => degradation; maintenance => all ok
 
## X is chi[3,t]*HI[t]
# 0implies03 {t in 1..99}: X[t] <= 100*chi[3,t]; # If chi = 0, X = 0
# ubound3 {t in 1..99}: X[t] <= HI[t]; # X <= HI, regardless of chi
# lbound3 {t in 1..99}: X[t] >= HI[t] - (1-chi[3,t])*100; # chi is 1, X = HI
 #small_maintenance {t in 1..99}: HI[t+1] <= 0.9*HI[t] - 0.9*Z[t] + Z[t] + 15*chi[1,t] + Y[t] + 70*chi[2,t]; # no maintenance => degradation; maintenance => all ok
 