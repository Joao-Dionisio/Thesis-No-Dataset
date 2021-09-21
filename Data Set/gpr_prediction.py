from adding_maintenance import read_excel
from data_analysis import data_cleaning
from gpr_plot import gpr_plot

import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C # don't forget matérn
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic

import pandas as pd

def gpr_prediction(kernel):
    #df, df3 = read_excel()
    #df = data_cleaning(df)
    #df = df[df["Ano Avaria"] != -9999]
    df = pd.read_csv("Dados_PATH.csv")
    return df
    #chosen = df["SAP ID"].mode()
    #df = df[df["SAP ID"] == chosen]
    df = df[df["SAPID"] == '280182379']
    df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    
    #df[["H2"]] = df[["H2"]].apply(pd.to_numeric, errors="coerce")
    df["Data Colheita"] = (df["Data Colheita"]-df["Data Colheita"][1930]).dt.days
    df = df[df['2FAL'].notna()]
    df["DP"] = (1.51 - np.log10(df["2FAL"]))/0.0035

    '''
    from matplotlib import pyplot as plt
    plt.plot(df["Data Colheita"], df["DP"])
    plt.show()
    '''
    
    X = df["Data Colheita"]
    Y = df["2FAL"]

    X = np.array(X).reshape(-1, 1)
    #Y = np.array(Y).reshape(-1, 1)

    #kernel  = C(1.0, (1e-3, 1e3))*RBF(10, (1, 1000))
    return df
    
    
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
    for i in range(len(X)):
        gpr.fit([X[i]], [Y.iloc[i]])
        
        test_features = np.atleast_2d(np.linspace(0,9000,100)).T

        y_pred , sigma = gpr.predict(test_features, return_std=True)

        gpr_plot(X[:i], Y[:i], test_features, y_pred, sigma)


def gpr_prediction(kernel):
    '''
    In order to do an ensemble, we can't work with too many gp's, so we only look at the data that has more than x observations
    '''
    df = pd.read_csv("Dados_PATH.csv")

    df = df[['SAPID', 'Data Colheita', '2FAL', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2',
              'O2', 'N2', 'COR', 'Massa_Volumica', 'Tensao_Interfacial', 'Tensao_Disruptiva',
              'Indice_Acidez', 'Teor_Agua', 'Tangente_Delta_90']]
    df = df[['SAPID', '2FAL', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2',
              'O2', 'Indice_Acidez', 'Teor_Agua']]

    #df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    #df.sort_values(by="Data Colheita")
    df.fillna(df.mean(), inplace=True)

    temp = df

    x = df["SAPID"].value_counts()
    x = x[x==25]
    temp = df[df["SAPID"] == x.index[0]]
    #temp["Data Colheita"] = (temp["Data Colheita"] - temp.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
    X_test, Y_test = temp.drop("2FAL", axis=1), temp["2FAL"] 
    
    # Get PTs with more than k(=25) records
    x = df["SAPID"].value_counts()
    x = x[x>25] # arbitrary number
    x = x.index

    df = df[df["SAPID"].isin(x)]
    #df["Data Colheita"] = (df["Data Colheita"] - df.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
    features, label = df.drop("2FAL", axis=1), df["2FAL"]
    gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
    gpr.fit(features, label)
    prediction, std = gpr.predict(X_test, return_std=True)
    
    print(Y_test)
    print(prediction)
        
    return grouped





def ensemble_prediction(kernel):
    '''
    In order to do an ensemble, we can't work with too many gp's, so we only look at the data that has more than x observations
    '''
    df = pd.read_csv("Dados_PATH.csv")

    df = df[['SAPID', 'Data Colheita', '2FAL', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2',
              'O2', 'N2', 'COR', 'Massa_Volumica', 'Tensao_Interfacial', 'Tensao_Disruptiva',
              'Indice_Acidez', 'Teor_Agua', 'Tangente_Delta_90']]
    df = df[['SAPID', 'Data Colheita', '2FAL', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2',
              'O2', 'Indice_Acidez', 'Teor_Agua']]

    df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    df.sort_values(by="Data Colheita")
    df.fillna(df.mean(), inplace=True)
    print(len(df))
    temp = df

    x = df["SAPID"].value_counts()
    x = x[x==25]
    temp = df[df["SAPID"] == x.index[0]]
    temp["Data Colheita"] = (temp["Data Colheita"] - temp.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
    X_test, Y_test = temp.drop("2FAL", axis=1), temp["2FAL"] 
    
    # Get PTs with more than k(=25) records
    x = df["SAPID"].value_counts()
    x = x[x>25] # arbitrary number
    x = x.index

    df = df[df["SAPID"].isin(x)]

        
    grouped = df.groupby(df["SAPID"])

    models  = []
    print(len(grouped))
    index = 0
    for PT in grouped:
        print(index)
        index+=1
        PT = pd.DataFrame(list(PT)[1]) # Convoluted way of converting to data frame
        PT["Data Colheita"] = (PT["Data Colheita"] - PT.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
        PT = PT[PT['2FAL'].notna()]
        features, label = PT.drop("2FAL", axis=1), PT["2FAL"] 
                
        gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
        gpr.fit(features, label)
        models.append(gpr)

        
    y_hat   = []
    weights = []
    for model in models:
        prediction, std = model.predict(X_test, return_std=True)
        y_hat.append(prediction)
        weights.append(std)

    # Doing a linear combination where the weights are proportional to how certain of a prediction they are
    weights = [1/i for i in weights] # Can you have 0 std?
    weights = [i/sum(weights) for i in weights]     
    
    #model_prediction = []
    #for i in range(len(y_hat[0])):
        #model_prediction.append(np.dot(y_hat[0], weights[0]))
    z = np.multiply(y_hat, weights)
    model_prediction = z.sum(axis=0)
    #model_prediction = np.dot(y_hat[0], weights[0])
    print(Y_test)
    print(model_prediction)
    return grouped

def clear_df():
    df = pd.read_csv("Dados_PATH.csv")

    df = df[['SAPID', '2FAL', 'Data Colheita']]
    
    df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    df.sort_values(by="Data Colheita")
    df.fillna(df.mean(), inplace=True)
    print(len(df))
    temp = df

    x = df["SAPID"].value_counts()
    x = x[x==25]
    temp = df[df["SAPID"] == x.index[0]]
    temp["Data Colheita"] = (temp["Data Colheita"] - temp.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
    X_test, Y_test = temp.drop("2FAL", axis=1), temp["2FAL"] 
    
    # Get PTs with more than k(=25) records
    x = df["SAPID"].value_counts()
    x = x[x>25] # arbitrary number
    x = x.index

    df = df[df["SAPID"].isin(x)]
    return df
        
from time import time as t

if __name__ == "__main__":
    #df = clear_df()
    kernel = RationalQuadratic(length_scale_bounds=(0.1, 200)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1))
    #gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    #gpr.fit(df["Data Colheita"], df["2FAL"])
    #gpr_plot()
    #grouped = ensemble_prediction(kernel)
    df = gpr_prediction(kernel)
