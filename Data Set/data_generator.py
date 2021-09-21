# Data Generator
'''
Due to the unavailability of the data, we had to generate our own


Plan:
    - Get an equation to model oil condition <- might be difficult
    - Plan oil replacement based on its condition, maybe introduce some randomness


After we have more data:
    - Try to the same thing that we already did
    
'''


'''
It stays about constant for a few years, and then it starts slowly increasing. 



Generate data for each PT individually
'''
import random
import math
import numpy as np

class component:
    def __init__(self, name, domain, inflection, steepness):
        self.name = name
        self.domain = np.linspace(0,domain,domain*100)
        self.inflection = inflection
        self.steepness = steepness

    # Assumig age of 100 years
    def health_simulation(self):
        '''
        h0 = 100
        aging_coeficient = 0.9
        return [h0*math.exp(i) for i in range(100)]
        '''
        #domain = range(30)
        #domain = np.linspace(0,30,1000)
        #y = [100*(1- 1 / (1 + math.exp(-(x-15)/3))) for x in domain]
        y = [100*(1- 1 / (1 + math.exp(-(x-self.inflection)/self.steepness))) for x in self.domain]
        self.health = y
        return [self.domain,y]

# Assumig age of 100 years
def dgaf_simulation():
    constant_time = random.randint(4,10)
    increasing_time = random.randint(3,6)
    surge_time = random.randint(1,3)

    dgaf = []
    
    for i in range(constant_time):
        dgaf.append(max(random.random()*3,1))

    for i in range(increasing_time):
        dgaf.append(max(min(random.random()*2.5*dgaf[-1],2000),1))
        
    for i in range(surge_time):
        dgaf.append(min(random.random()*15*dgaf[-1],10000))
    return dgaf

def simulate_gases():
    # DGA thresholds from Towards a comprehensive health index

    Htwo = []#[100,200,300,500,700]
    CHfour = []#[75,125,200,400,600]
    CtwoHsix = []#[65,80,100,120,150]
    CtwoHfour = []#[50,80,100,150,200]
    CtwoHtwo = []#[3,7,35,50,80]
    CO = []#[350,700,900,1100,1400]
    COtwo = []#[2500,3000,4000,5000,7000]

    
    bolts = component('bolts',40,15,10)
    oil_preservation_system = component('OPS',30,15,10)
    joints = component('Joints',10,4,4)

    bolts.health_simulation()
    oil_preservation_system.health_simulation()
    joints.health_simulation()

    #gases=[H2,CH4,C2H2,C2H4,C2H6,CO,CO2]
    

    # We're going to assume that the deterioration of the components leads to an increase of the gases indiscriminantly
    # the failure of different components leads to different increases in dga gases
    # maybe have the gases increase exponentially independent of the components, but these remove some of it
    dga_values = [] 
    for i in range(15):
        pt_health = (bolts.health[i]+oil_preservation_system.health[i]+joints.health[i])/3

        Htwo.append(math.exp(i)/(25*pt_health))
        CHfour.append(math.exp(i)/(33*pt_health))
        CtwoHtwo.append(math.exp(i)/(38*pt_health))
        CtwoHfour.append(math.exp(i)/(50*pt_health))
        CtwoHsix.append(math.exp(i)/(833*pt_health))
        CO.append(math.exp(i)/(7*pt_health))
        COtwo.append(math.exp(i)/pt_health)
        
        '''
        Htwo.append(math.exp(i/(0.025*pt_health)))
        CHfour.append(math.exp(i/(33*pt_health)))
        CtwoHtwo.append(math.exp(i/(38*pt_health)))
        CtwoHfour.append(math.exp(i/(50*pt_health)))
        CtwoHsix.append(math.exp(i/(833*pt_health)))
        CO.append(math.exp(i/(7*pt_health)))
        COtwo.append(math.exp(i/pt_health))
        '''
        
        #dga_values.append([cur_Htwo,cur_CHfour,cur_CtwoHtwo,cur_CtwoHfour,cur_CtwoHsix,cur_CO,cur_COtwo])

    return [Htwo,CHfour,CtwoHsix,CtwoHfour,CtwoHtwo,CO,COtwo]




'''
# Assumig age of 100 years
def health_simulation(component):
    h0 = 100
    aging_coeficient = 0.9
    return [h0*math.exp(i) for i in range(100)]
    #domain = range(30)
    #domain = np.linspace(0,30,1000)
    #y = [100*(1- 1 / (1 + math.exp(-(x-15)/3))) for x in domain]
    y = [100*(1- 1 / (1 + math.exp(-(x-component.inflection)/component.steepness))) for x in component.domain]
    return [component.domain,y]
'''
    
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    #print(oil_health_simulation())

    #oil = component(30,15,3)
    bolts = component('bolts',40,15,10)
    cooling_system = component('Cooling System',30,15,15)
    oil_preservation_system = component('OPS',30,15,10)
    bearing = component('Bearings',30,15,8)
    joints = component('Joints',10,4,4)

    base_components = [bolts, cooling_system, oil_preservation_system, bearing, joints]    
    colors = ['red', 'green', 'yellow', 'blue', 'pink']
    for index, cur_component in enumerate(base_components):
        #x,y = health_simulation(cur_component)
        x,y = cur_component.health_simulation()
        plt.scatter(x,y,s=10,color=colors[index])

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Bolts', markerfacecolor='red', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Cooling System', markerfacecolor='green', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='OPS', markerfacecolor='yellow', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Bearings', markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Joints', markerfacecolor='pink', markersize=15)]
    plt.legend(handles=legend_elements, prop={'size': 10}, markerscale=0.6)
    plt.show()
    plt.clf()
    
    x = simulate_gases()
    for j in x[:-1]:
        plt.scatter([i for i in range(15)],j[:15])
    plt.show()
