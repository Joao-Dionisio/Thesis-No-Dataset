import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic
import numpy as np
from random import random


# object for keeping instance data
class INST:
    pass
inst = INST()
inst.t = 1    # time for probing
inst.s = 1    # traveling speed
inst.T = 100  # time limit for a route
inst.x0 = 0   # depot coordinates
inst.y0 = 0   # depot coordinates

# auxiliary function: euclidean distance
def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Auxiliary function: return sum of distances between a point and a set of points.
def min_distance(x1,y1,points):
    return sum([dist(x1,y1,x2,y2) for x2,y2 in points])

# find max variance in a grid
def max_var(gpr, n):
    # prepare data
    x = np.arange(0, 1.01, 1./n)
    y = np.arange(0, 1.01, 1./n)
    X = [(x[i],y[j]) for i in range(n+1) for j in range(n+1)]
    z, z_std = gpr.predict(X,return_std=True)
    points = [(x,v) for (v,x) in sorted(zip(zip(z_std,z),X))]
    # print(z_std)
    # print(X)
    # for i in range(len(X)):
    #     print("\t{}\t{}\t{}".format(i,X[i],z_std[i]))
    return points


def search_around_point(x,y,pos,gpr, n=10):
    # Given a valuable cell on a grid, searches for random points near it that have bigger variance or are
    # closest to the other, already fixed, points

    possible_points = [(x,y)]+[(x + (2*random()-1)/100, y + (2*random()-1)/100) for _ in range(n)]

    z, z_std = gpr.predict(possible_points,return_std=True)
    points = [(x,v) for (v,x) in sorted(zip(zip(z_std,z),possible_points))]
    
    best_z, best_points = (z_std[0], z[0]), (x,y)
    ######################################
    # Initializing the variables
    best_distance = min_distance(best_points[0], best_points[1], pos)
    improv = True
    while improv == True:
        improv = False
        for index, cur_std in enumerate(z_std):
            cur_z = (z_std[index], z[index])
            cur_coor = (possible_points[index][0], possible_points[index][1])
            if cur_std >= best_z[0]:
                temp_distance = min_distance(cur_coor[0], cur_coor[1], pos)
                if temp_distance < best_distance:
                    best_distance, best_z, best_points = temp_distance, cur_z, cur_coor
                    improv = True
            elif cur_std > best_z[0]:
                best_z = cur_z
                best_points = possible_points[index]
                improv = True
            
        if improv == True:
            possible_points = [best_points]+[(best_points[0] + (2*random()-1)/100, best_points[1] + (2*random()-1)/100) for _ in range(n)]
            z, z_std = gpr.predict(possible_points,return_std=True)
    #######################################
    print('found point nº ', len(pos))
    return (best_points, best_z)

# required function: route planning
def planner(X,z):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
        - 
    """
    X = list(X) # work with local copies
    z = list(z)
    from tsp import solve_tsp, sequence  # exact
    from tsp import multistart_localsearch  # heuristic
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # # plot preliminary GP
    # from functions import plot
    # from functions import f1
    # plot(f1,100)
    # # end of plot
    # # # plot posteriori GP
    # GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # GPR.fit(X, z)
    # def GP(x_,y_):
    #     return GPR.predict([(x_,y_)])[0]
    # plot(GP,100)
    # # # end of plot

    x0, y0 = (inst.x0,inst.y0)  # initial position (depot)
    pos = [(x0,y0)]
    N = 1
    c = {}
    while True:
        # attempt adding Nth "city" to the tour
        points = max_var(gpr, n=100)  # n x n grid !!!!!
        (x, y), (z_new_std, z_new) = points.pop()
        while (x,y) in pos:  # discard duplicates
            (x, y), (z_new_std, z_new) = points.pop()

        # Now that we have the candidate cell, we can do the random search
        #(x,y), (z_new_std, z_new) = search_around_point(x,y,pos,gpr)
        
        for i in range(N):
            c[i,N] = dist(*pos[i], *(x,y))
            c[N,i] = c[i,N]

        if N < 3:
            N += 1
            pos.append((x,y))
            continue # !!!!!

        sol_, obj = multistart_localsearch(100, N+1, c, cutoff=inst.T - (N)*inst.t);  # heuristic
        if obj <= inst.T - (N)*inst.t:
            idx = sol_.index(0); sol = sol_[idx:] + sol_[:idx]  # heuristic
            # print(obj + (N)*inst.t, "TSP solution:", obj, N, inst.T, sol)
            # print("appending", (x,y), z_new_std, z_new, "orient.len:", obj + (N)*inst.t)
            N += 1
            assert (x,y) not in pos
            pos.append((x,y))
            X.append((x,y))
            z.append(z_new)  # !!!!! with average of the prediction as an estimate
            gpr.fit(X, z)
        else:
            # attempt exact solution:
            # print("attempting exact solution")
            obj, edges = solve_tsp(range(N+1), c)  # exact
            print(obj+N*inst.t)
            print(inst.T)
            if obj <= inst.T - (N) * inst.t:
                sol = sequence(range(N + 1), edges)              # exact
                # print(obj + (N) * inst.t, "TSP EXACT:", obj, N, inst.T, sol)
                # print("appending", (x, y), z_new_std, z_new, "orient.len:", obj + (N) * inst.t)
                N += 1
                pos.append((x, y))
                X.append((x, y))
                z.append(z_new)  # !!!!! with average of the prediction as an estimate
                gpr.fit(X, z)
                continue
            break 

    print(obj + (N)*inst.t, "TSP solution:", obj, N, inst.T, sol)
    return [pos[i] for i in sol[1:]]


# required function: route planning with probing
def explorer(X,z,plan):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
        - plan: list of coordinates of initial probing plan [(x1,y1), ...., (xP,yP)]
                this plan may be changed; the only important movement is (x1,y1)
    """
    X = list(X) # work with local copies
    z = list(z)
    from tsp import solve_tsp, sequence
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

    x0, y0 = (inst.x0,inst.y0)  # initial position (depot)
    x1, y1 = plan[0]
    pos = [(x0,y0), (x1,y1)]
    N = 2
    c = {}
    c[0, 1] = dist(*pos[0], *pos[1])
    while True:
        # attempt adding Nth "city" to the tour
        points = max_var(gpr, n=100)  # n x n grid !!!!!
        (x, y), (z_new_std, z_new) = points.pop()
        for i in range(N):
            c[i,N] = dist(*pos[i], *(x,y))

        if N < 3:
            N += 1
            pos.append((x,y))
            continue # !!!!!
        obj, edges = solve_tsp(range(N+1), c)
        if obj <= inst.T - (N)*inst.t:
            sol = sequence(range(N+1), edges)
            print("TSP solution:", obj, N, inst.T, sol)
            print("appending", (x,y), z_new_std, z_new)
            N += 1
            pos.append((x,y))
            X.append((x,y))
            z.append(z_new)  # !!!!! with average of the prediction as an estimate
            gpr.fit(X, z)
        else:
            # maybe some other elements of 'points' should be attempted here
            break

    return [pos[i] for i in sol[1:]]


def estimator(X,z,mesh):
    """estimator: evaluate z at points [(x1,y1), ...., (xK,yK)] based on M=n+N known points:
        - X: list of coordinates [(x1,y1), ...., (xM,yM)]
        - z: list of evaluations of "true" function [z1, ..., zM]
    """
    # from tools import trainGP
    # gpr = trainGP(X, z)

    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    gpr.fit(X, z)

    z = []
    for (x_,y_) in mesh:
        val = gpr.predict([(x_,y_)])
        z.append(val)

    return z
