def h1_increasing_load(t,K):
    import math
    return 0 + (2*1.4*14.5*(K**1.3)-0)*(1-math.e**((-t)/(2*7)))

def h2_increasing_load(t,K):
    import math
    return 0 + ((2-1)*1.4*14.5*(K**1.3)-0)*(1-math.e**((-t)/(150/2)))

def hs_temp_diff(t,K):
    return h1_increasing_load(t,K)-h2_increasing_load(t,K)

def final_temp(t,K):
    import math
    return 23.9 + 38.3*((1+1000*K**2)/(1+1000))**0.8 - 38.3*((1+1000*K**2)/(1+1000))**0.8*math.e**(-t/(0.5*210))+ hs_temp_diff(t,K)


