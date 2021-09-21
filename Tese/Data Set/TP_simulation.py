# Trying to simulate a Power Transformer (PT) throughout its life cycle

'''
Best Kernels

ConstantKernel(1) -> 6.27%

'''



import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic

from read import read_file
from start import split_data, calculate_error, normalize

kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))

constant = 0.001
best = float('inf')
while constant < 100000:
    np.random.seed(0)
    #kernel = 1.0 * RBF(constant)
    #kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    kernel = RationalQuadratic(length_scale_bounds=(4.1, 8)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    #kernel = ConstantKernel(constant)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    info = read_file()

    train, test, train_health, test_health = split_data(info)

    train = np.array(train)
    test = np.array(test)
    train = normalize(train)
    test = normalize(test)

    test_health = [i/max(test_health) for i in test_health]
    train_health = [i/max(train_health) for i in train_health]


    gpr.fit(train, train_health)
    predicted_health = gpr.predict(test)

    a = calculate_error(predicted_health, test_health)
    best = min(best, a)
    print(best/sum(test_health))
    constant*=2