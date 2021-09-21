param Tmax;                     # upper bound on the number of periods
set components = {"Bolts", "Bearing", "Gasket", "Cooling System", "OPS", "Capacitance Tap", "Oil Impregnated Paper", "OLTC", "Bushing", "Oil", "Winding"};   # separate maintenance items
param RULmax{k in components};                   # initial remaining useful life for each item k in K
param Cost{k in components};                      # fixed cost for replacement of k in K
param Price;                              # unit price (of electricity?)
param Qmax;                           # production capacity
param d;                              # equipment aging factor
param load_effect;                              # equipment load usage factor
param periodicity;					  # periodicity of preventive maintenance
set failure_time;

# for testing, not actually used here
param best;
param index_best;

var q{1..Tmax}>=0;                     # quantity produced/sold in period t
var qmax{1..Tmax}>=0;                  # adjusted capacity for period t
var rul{k in components,0..Tmax} >= 0, <= RULmax[k];   # remaining useful life for item k in components
var x{k in components,1..Tmax} binary;            # maintenance/replacement of parts k in components
var force_shutdown{1..Tmax} binary;

maximize profit:
    sum {t in 1..Tmax} (Price*q[t]-sum{k in components} Cost[k]*x[k,t]);

# Assuming PT is brand new
RUL0{k in components}: 
    #rul[k,0] = RULmax[k]; 
    rul[k,0] = 0; # <- takes into consideration if it's worth it to buy an equipement

RUL_bolts{t in 1..Tmax}: 
    rul["Bolts",t] <=  rul["Bolts",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["Bolts"]*x["Bolts",t];

RUL_bearing{t in 1..Tmax}: 
    rul["Bearing",t] <=  rul["Bearing",t-1]*d  - load_effect*q[t]/Qmax + 2*RULmax["Bearing"]*x["Bearing",t];

RUL_OPS{t in 1..Tmax}: 
    rul["OPS",t] <=  rul["OPS",t-1]*d  - load_effect*q[t]/Qmax - load_effect*rul["Gasket",t]/RULmax["Gasket"] + 2*RULmax["OPS"]*x["OPS",t];

RUL_gasket{t in 1..Tmax}: 
    rul["Gasket",t] <=  rul["Gasket",t-1]*d  - load_effect*rul["Bolts",t]/RULmax["Bolts"] + 3*RULmax["Gasket"]*x["Gasket",t]

RUL_Cooling_System{t in 1..Tmax}: 
    rul["Cooling System",t] <=  rul["Cooling System",t-1]*d - load_effect*rul["Bearing",t]/RULmax["Bearing"] + 3*RULmax["Cooling System"]*x["Cooling System",t]

RUL_oil{t in 1..Tmax}: 
    rul["Oil",t] <=  rul["Oil",t-1]*d  - load_effect*rul["OPS",t]/RULmax["OPS"] - load_effect*rul["Winding",t]/RULmax["Winding"] + 4*RULmax["Oil"]*x["Oil",t]

RUL_capacitance_tap{t in 1..Tmax}: 
    rul["Capacitance Tap",t] <=  rul["Capacitance Tap",t-1]*d  - load_effect*q[t]/Qmax - load_effect*rul["Gasket",t]/RULmax["Gasket"] + 3*RULmax["Capacitance Tap"]*x["Capacitance Tap",t];

RUL_paper{t in 1..Tmax}: 
    rul["Oil Impregnated Paper",t] <=  rul["Oil Impregnated Paper",t-1]*d - load_effect*rul["Gasket",t]/RULmax["Gasket"] - load_effect*rul["Oil",t]/RULmax["Oil"] - load_effect*q[t]/Qmax + 5*RULmax["Oil Impregnated Paper"]*x["Oil Impregnated Paper",t];

RUL_bushing{t in 1..Tmax}: 
    rul["Bushig",t] <=  rul["Bushing",t-1]*d - load_effect*rul["Capacitance Tap",t]/RULmax["Capacitance Tap"] - load_effect*rul["Oil Impregated Paper",t]/RULmax["Oil Impregnated Paper"] - load_effect*q[t]/Qmax + 5*RULmax["Bushig"]*x["Bushig",t];

RUL_OLTC{t in 1..Tmax}: 
    rul["OLTC",t] <=  rul["OLTC",t-1]*d - load_effect*rul["Cooling System",t]/RULmax["Cooling System"] - load_effect*q[t]/Qmax + 3*RULmax["OLTC"]*x["OLTC",t];

RUL_winding{t in 1..Tmax}:     
    rul["Winding",t] <= rul["Winding",t-1]*d - load_effect*q[t]/Qmax - load_effect*rul["Cooling System",t]/RULmax["Cooling System"] - load_effect*rul["Oil",t]/RULmax["Oil"] + 4*RULmax["winding"]*x["winding",t] - force_shutdown[t]*1; # constant term has to disappear with shutdown, otherwise maintenance is forced

# Adjusting capacity for aging equipment
QMAX{k in components, t in 1..Tmax}: 
    qmax[t] <= Qmax * rul[k,t]/RULmax[k];

Q{k in components, t in 1..Tmax}: 
    q[t] <= qmax[t];


# Forcing smaller maintenances to occur when larger is made
oil_implies_joints{t in 1..Tmax}: 
	x["oil", t] <= x["joints", t];

winding_implies_oil{t in 1..Tmax}: 
	x["winding", t] <= x["oil",t]; # winding implies joints is above

bushing_implies_tap{t in 1..Tmax}: 
	x["Bushing", t] <= x["Capacitance Tap",t]; # winding implies joints is above


everything_stops{t in 1..Tmax, k in components}: 
	rul[k,t] <= force_shutdown[t]*RULmax[k];


stopping_is_definitive{t in 1..Tmax-1}: 
	force_shutdown[t+1] <= force_shutdown[t];
