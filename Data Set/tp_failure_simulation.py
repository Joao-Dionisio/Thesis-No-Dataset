# Trying to simulate PT failure according to Luis Guimarães' paper

import numpy as np

failure_treshholds_mean = {
    "TDCG":2315,
    "H_2":900,
    "CH_4":500,
    "C_2H_2":17,
    "C_2H_4":100,
    "C_2H_6":75,
    "CO":700,
    "CO_2":5000
}

def failure_classification(DGA):
    failure_treshholds = [
        [720,100,120,1,50,65,350,2500],
        [1920,700,400,9,100,100,570,4000],
        [4630,1800,1000,35,200,150,1400,10000]
    ]
        
    for fault in range(3):
        if all(DGA[i] <= failure_treshholds[fault][i] for i in range(8)):
            return fault+1
    return 4


def generate():
    return [np.random.normal(failure_treshholds_mean[i],100) for i in failure_treshholds_mean]

for _ in range(10000):
    x = failure_classification(generate())
    if x == 1:
        print(x)

