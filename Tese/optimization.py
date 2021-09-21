class PT:
    def __init__(self, t0, H0, minor_maintenance_duration, major_maintenance_duration, minor_maintenance_cost, major_maintenance_cost, replacement_cost):
        self.t = t0 # 30 minute intervals?
        self.HI = H0
        self.minor_maintenance_duration = minor_maintenance_duration
        self.major_maintenance_duration = major_maintenance_duration
        self.minor_maintenance_cost = minor_maintenance_cost
        self.major_maintenance_cost = major_maintenance_cost
        self.replacement_cost = replacement_cost
        self.cost = 0
        return
    
    def apply_minor_maintenance(self):
        minor_maintenance_duration = self.calculate_minor_maintenance_duration()
        self.downtime.extend(range(self.t, self.t+self.minor_maintenance_duration+1))
        self.cost+=self.minor_maintenance_cost
        self.HI = some_minor_function(self.HI)
        return

    def apply_major_maintenance(self):
        minor_maintenance_duration = self.calculate_major_maintenance_duration()
        self.downtime.extend(range(self.t, self.t+self.minor_maintenance_duration+1))
        self.cost+=self.major_maintenance_cost
        self.HI = some_major_function(self.HI)
        return

    def calculate_minor_maintenance_duration(self):
        return constant + random

    def calculate_major_maintenance_duration(self):
        return constant + random
        
    def calculate_downtime(self):
        self.downtime = list(set(range(t0, t_final+1)) - set(self.downtime))
        return

    def replace_asset(self):
        self.cost+=self.replacement_cost
        self.HI = 100
        return

    def calculate_revenue(self):
        self.calculate_downtime() # we need to separate minor from major to associate costs
        costs = 0
        for i in range(self.t0+1, self.t_final+1):
            if i in self.downtime:
                pass

        return

def expected_annual_cost(minor_fault_rate, minor_maintenance_cost, major_fault_rate, major_maintenance_cost, n):
    cost = minor_fault_rate[n]*minor_maintenance_cost + major_fault_rate[n]*major_maintenance_cost
    return cost

def calculate_EUAC(t0, t_final, minor_fault_rate, minor_maintenance_cost, major_fault_rate, major_maintenance_cost, plot=False):
    from matplotlib import pyplot as plt
    total_cost = 0
    EUAC = [0]
    for i in range(t0+1, t_final+1):
        cost = expected_annual_cost(minor_fault_rate, minor_maintenance_cost, major_fault_rate, major_maintenance_cost, i)
        total_cost+=cost
        EUAC.append(total_cost/(i-t0)) 
    if plot:
        plt.scatter(EUAC)
    return min(EUAC)