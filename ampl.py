from amplpy import AMPL, Environment, DataFrame
import random

# The .mod files are in the right place, ampl_mswin64, where ampl is installed
def main():
    #ampl = AMPL(Environment(....))
    ampl = AMPL()
    ampl.setOption('solver','gurobi')
    ampl.setOption('gurobi_options','timelim=1')

    # Reading Model and Data
    #ampl.read("1-6.mod")
    #ampl.readData("scenario1.dat")
    ampl.read("basemodel3.mod")
    #ampl.read("model.mod")
    #ampl.readData("basemodel.dat")
    
    
    # Change value of t_final to 30
    #time = ampl.getParameter('t_final')
    #time.setValues([30])
    time = ampl.getParameter('Tmax')

    #min_health = ampl.getValue('min_health')

    #min_health = ampl.getValues().toList()[0]

    failure = ampl.getSet('failure_time')
    
    # Assuming that failure is independant of health 
    failures = []
    for i in range(1,int(time.value())+1):
        if random.random() < 0.07:
            failures.append(i) # failure rate of 0.07/year in normal region -> Techno-Economic Method   

    print(failures)
    failure.setValues(failures) # <- param failures created in .mod, need to find out how to feed the values
    print(ampl.getSet('failure_time').getValues())
    #return failure.getValues().toList()
    ampl.solve()

    #ampl.getObjective('cost').value()
    ampl.getObjective('profit').value() 
    #print(ampl.getVariable('HI').getValues()) # returns data frame getValues().toList()
    print(ampl.getVariable('rul').getValues()) # returns data frame getValues().toList()
    return ampl

a = main()
