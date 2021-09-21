import math
import random

# object for keeping instance data
class INST:
    pass
inst = INST()
inst.t = 100    # time for probing
inst.s = 1    # traveling speed
inst.T = 1  # time limit for a route
inst.x0 = 0   # depot coordinates
inst.y0 = 0   # depot coordinates

EPS = 1.e-6   # for floating point comparisons
INF = float('Inf')

# auxiliary function: euclidean distance
def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def evaluate(f, planner, estimator):
    # prepare data
    X, z = [], []
    N = 5
    
    # grid
    for x in [i/N for i in range(1,N)]:
        for y in [i/N for i in range(1,N)]:
            X.append((x,y))
            z.append(f(x,y))
    '''
    # alternative to grid: random points
    for x in [i/N for i in range(1,N)]:
         for y in [i/N for i in range(1,N)]:
             x = random.random()
             y = random.random()
             X.append((x,y))
             z.append(f(x,y))
             print("{}\t&{}\t&{}\\\\".format(x,y,z[-1]))
    '''
    # test preliminary forecasting part
    mesh = []
    for i in range(101):
        for j in range(101):
            x, y = i/100., j/100.
            mesh.append((x,y))
    z0 = estimator(X,z,mesh)
    prelim = 0
    for i in range(len(mesh)):
        (x, y) = mesh[i]
        prelim += abs(f(x,y) - float(z0[i]))

    # test planning part
    route = planner(X,z)
    tsp_len = 0     # elapsed time
    (xt,yt) = (inst.x0,inst.y0)
    for (x,y) in route:
        tsp_len += dist(xt, yt, x, y) / inst.s + inst.t
        xt, yt = x, y
        X.append((x,y))
        z.append(f(x, y))
        print("probing at ({:8.5g},{:8.5g}) --> \t{:8.5g}".format(x, y, z[-1]))

    # # # plot posteriori GP
    # from sklearn.gaussian_process import GaussianProcessRegressor
    # from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic
    # kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    # from functions import plot
    # GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # GPR.fit(X, z)
    # def GP(x_,y_):
    #     return GPR.predict([(x_,y_)])[0]
    # plot(GP,100)
    # # # end of plot


    tsp_len += dist(xt, yt, inst.x0, inst.y0) / inst.s
    if tsp_len > inst.T + EPS:
        print("tour length infeasible:", tsp_len)
        # return INF    # route takes longer than time limit

    # test forecasting part
    mesh = []
    for i in range(101):
        for j in range(101):
            x, y = i/100., j/100.
            mesh.append((x,y))
    z = estimator(X,z,mesh) # estimation of information gained (?)
    final = 0
    for i in range(len(mesh)):
        (x, y) = mesh[i]
        final += abs(f(x,y) - float(z[i])) # Calculating the error (?)

    return prelim, tsp_len, final

        
if __name__ == "__main__":
    #from functions import f1 as f
    #from static import planner,estimator
    #random.seed(0)
    #prelim, tsp_len, final = evaluate(f, planner, estimator)
    #print("student's evaluation:\t{:8.7g}\t[TSP:{:8.7g}]\t{:8.7g}".format(prelim, tsp_len, final))

    from static import planner, estimator
    import functions as fs
    for f in [fs.f1,fs.f2,fs.f3,fs.f4,fs.f5,fs.f6,fs.f7,fs.f8,fs.f9,fs.f10]:
        result = []
        random.seed(0)
        prelim, tsp_len, final = evaluate(f, planner, estimator)
        print("student's evaluation:\t{:8.7g}\t[TSP:{:8.7g}]\t{:8.7g}".format(prelim, tsp_len, final))
        result.append(final)
    print(result)
        
    
    # # for reading trend from csv file:
    # import csv
    # import gzip
    # with gzip.open("data_2017_newsvendor.csv.gz", 'rt') as f:
    #     reader = csv.reader(f)
    #     data = [int(t) for (t,) in reader]
