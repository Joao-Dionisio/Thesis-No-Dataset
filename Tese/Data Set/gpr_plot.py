# Let's try to plot GP regression, make those cool little graphs
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def f(x):
    """The function to predict."""
    #return x * np.sin(x)# + x**(x/10)
    return 0.1*x**2 + np.sin(x) + 1
'''
def gpr_plot(training_features, training_labels, test_features, test_labels, sigma):

    X = training_features
    y = training_labels
    x = test_features
    y_pred = test_labels

    plt.plot(X, y, 'r.', markersize=10, label='Observations')    
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('Data Colheita')
    plt.ylabel('$2 FAL$')
    plt.ylim(-1, 3)
    plt.legend(loc='upper left')

    plt.show()
'''
def gpr_plot(training_features, training_labels, test_features, test_labels, sigma):

    X = training_features
    y = training_labels
    x = test_features
    y_pred = test_labels

    plt.plot(X, y, 'r.', markersize=10, label='Observations')    
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('GP Prediction')
    ymin = min(min(y), min(y_pred))
    ymax = max(max(y),max(y_pred))
    plt.ylim(ymin - abs(ymin/10) , ymax + abs(ymax/10))
    plt.legend(loc='upper left')

    mu = 9
    variance = 0.1
    sigma = math.sqrt(variance)
    x_ = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x_, stats.norm.pdf(x_,mu,sigma))
    #plt.plot(X)

    plt.show()


def gpr_plot0(test_features, test_labels, sigma):

    x = test_features
    y_pred = test_labels

    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('$x*sin(x)$')
    plt.ylim(-10, 10)
    plt.legend(loc='upper left')

    plt.show()


    
if __name__ == "__main__":
    import numpy as np

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C # don't forget matérn
    from sklearn.gaussian_process.kernels import RationalQuadratic

    
    np.random.seed(0)
    X = [i for i in [1,2,5,7,9]]
    #X = np.asarray([0.5,2,5,7,9], dtype=np.float32)
    #X = np.arange(0,10,0.1)
    #X = [1,2,3,5]
    #X = np.atleast_2d(np.linspace(0,10,100)).T
    Y = [f(i) for i in X]
    
    X = np.array(X).reshape(-1, 1)
    #Y = np.array(Y).reshape(-1, 1)

    #kernel  = C(1.0, (1e-3, 1e3))*RBF(10, (1e-2, 1e2))
    kernel = RationalQuadratic(length_scale=4, alpha=0.4)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    gpr.fit(X, Y)

    test_features = np.atleast_2d(np.linspace(0,10,1000)).T
    print(test_features)
    y_pred , sigma = gpr.predict(test_features, return_std=True)

    gpr_plot(X, Y, test_features, y_pred, sigma)
    
    #gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    #y_pred , sigma = gpr.predict(test_features, return_std=True)
    #gpr_plot0(test_features, y_pred, sigma)
