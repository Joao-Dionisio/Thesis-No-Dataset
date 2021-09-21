from amplpy import AMPL, Environment, DataFrame

def main():
    ampl = AMPL(Environment("C:/Users/jdion/AppData/Local/Programs/Python/Python37/Lib/site-packages"))
    ampl.option['solver'] = 'gurobi'
    

    ampl.read("1-6.mod")
    #ampl.readData("scenario1.dat")
    ampl.solve()
    return ampl

main()
