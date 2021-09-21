import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from read import read_file
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

def really():
    df = pd.read_csv("Dados_PATH.csv")
    df = df.apply(lambda x: x.astype(str).str.replace(',','.'))
    df["Data Colheita"] = pd.to_datetime(df["Data Colheita"])
    
    df[["SAPID","2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2','Teor_Agua']] = df[["SAPID","2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2','Teor_Agua']].apply(pd.to_numeric, errors='coerce')
    #df = df[df[["SAPID", "2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2', "Data Colheita", 'Diagnostico_FQ','Recomendacoes']].notna()]
    #df = df[df[["SAPID", 'Teor_Agua', "Data Colheita", 'Diagnostico_FQ','Recomendacoes']].notna()]
    df = df[["SAPID", 'Teor_Agua', "Data Colheita", 'Diagnostico_FQ','Recomendacoes']]
    df.dropna(how='any', inplace=True)

    #twoFAL = df[["SAPID", "2FAL", "Data Colheita"]]

    #DGA = df[["SAPID",'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2', 'Data Colheita']]

    df['Oil Condition'] = df.apply(lambda row: create_oil_condition(row), axis=1)
    df['Oil Regeneration'] = df.apply(lambda row: create_oil_regeneration(row), axis=1) # a regeneração ocorreu no momento anterior

    df['Oil_RUL'] = df.groupby(['SAPID','Oil Regeneration'])['Data Colheita'].apply(lambda x: (x.max() - x))    
    df['Oil_RUL?'] = df['Oil_RUL'].shift(-1)
    return df


def create_oil_condition(row):

    #if 'degradação avançada' in row['Diagnostico_FQ']:# and 'regeneração' in row['Recomendacoes']:
    #    return 3
    if 'degradação avançada' in row['Diagnostico_FQ']:
        return 2
    elif 'alguma degradação' in row['Diagnostico_FQ']:
        return 1
    else:
        return 0


def create_oil_regeneration(row):
    if 'regeneração' in row['Recomendacoes']:
        return 1
    else:
        return 0

def gp_predict_with_moisture(df):
    df.dropna(how='any', inplace=True)
    df = df[['Teor_Agua', 'Oil_RUL?']]
    
    X, y = df['Teor_Agua'].values, df['Oil_RUL?'].dt.days.values

    #return X, y
    #X = X.values.reshape(-1,1)
    X = X.reshape(-1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=73)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=10)
    
    gpr.fit(X_train, y_train)

    gpr.predict(X_test)
    y_pred = gpr.predict(X_test)
    print(r2_score(y_test, y_pred))
    
x = really()


a,b = gp_predict(x)
