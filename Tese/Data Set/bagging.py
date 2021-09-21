from numpy import mean
from numpy import std
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import BaggingRegressor
from gpr_plot import gpr_plot
import pandas as pd
import numpy as np

#X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=5)

#kernel = RationalQuadratic(length_scale_bounds=(0.1, 200)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1))
#model = BaggingRegressor(base_estimator=GaussianProcessRegressor())

class BaggingRegressor:
    def __init__(self, df, kernel, n_subsets):
        self.kernel = kernel
        self.df = df
        self.n_subsets = n_subsets
        #self.gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)

    def _split_data_no_replacement(self):
        n_subsets = self.n_subsets
        return [self.df.sample(len(self.df)//n_subsets) for i in range(n_subsets)]

    def _generate_models(self):
        subsets = self._split_data_no_replacement()
        X_train = []
        X_test  = []
        y_train = []
        y_test  = []
        models  = []
        for subset in subsets:
            features, label = subset[...], subset[...] 
            info = train_test_split(features, label, test_size=0.7)
            X_train.append(info[0])
            #X_test.append(info[1])
            Y_train.append(info[2])
            #Y_test.append(info[3])
            gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
            gpr.fit(X_train, Y_train)
            models.append(gpr)
        self.models = models
        return models

    
    def predict(self, X_test):
        try:
            models = self.models
        except:
            models = self._generate_models()

        y_hat = []
        weights = []
        for model in models:
            prediction, std = model.predict(X_test, return_std=True)
            y_hat.append(prediction)
            weights.append(std)

        weights = [1/i for i in weights] # Can you have 0 std?
        weights = [i/sum(weights) for i in weights]     

        model_prediction = np.dot(y_hat, weights) # dot product 
        
        return model_prediction
    
    
df = pd.read_csv("Dados_PATH.csv")
df = df[df["SAPID"] == '280182379']
df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')

df["Data Colheita"] = (df["Data Colheita"]-df["Data Colheita"][1930]).dt.days
df = df[df['2FAL'].notna()]
df["DP"] = (1.51 - np.log10(df["2FAL"]))/0.0035

X = df["Data Colheita"]
Y = df["2FAL"]

X = np.array(X).reshape(-1,1)

model.fit(X, Y)

test_features = np.atleast_2d(np.linspace(0,9000,100)).T

y_pred, sigma = model.predict(test_features, return_std=True)

gpr_plot(X, Y, test_features, y_pred, sigma)

'''
model.fit(X, Y)
row = [[0.88950817,-0.93540416,0.08392824,0.26438806,-0.52828711,-1.21102238,-0.4499934,1.47392391,-0.19737726,-0.22252503,0.02307668,0.26953276,0.03572757,-0.51606983,-0.39937452,1.8121736,-0.00775917,-0.02514283,-0.76089365,1.58692212]]
yhat = model.predict(row)

print('Prediction: %d' % yhat[0])

model = BaggingRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
'''
