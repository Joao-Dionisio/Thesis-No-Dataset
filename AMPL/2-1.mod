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

var hi;
var t;
var load;


maximize RUL: t;



subject to
ends_at_t: HI[t] == 0;
works_until_t {i in 0..t-1}: HI[i] > 0;

one_decision: product {t in 0..100} sum {i in 0..1} chi[i][t] == 1 # Ensuring one decision at every point in time

# Assuming full repair and one maintenance
small_improv:   chi_1 = 1 ==> HI = 100;
#small_downtime: chi_1 = 1 ==> R(t') = 0, forall t' in t_0..t_0+delta_1;



#energy_requirement {t in 0..100}: load[t] >= 50; # assuming constant demand
#energy_capacity {t in 0..100}:    load[t] <= max_capacity[HI];