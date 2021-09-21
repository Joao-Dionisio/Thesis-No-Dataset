# Trying to estimate dga depreciation

# Look at Towards a Comprehensive DGA Health Index

# Script to remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C # don't forget matérn
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic
from data_analysis import really

from sklearn.linear_model import LinearRegression

def predictDGA_bagging(kernel):
    _, df = really()
    #df["DGAF"] = (2*df['H2']+3*df['CH4']+5*df['C2H2']+3*df['C2H4']+3*df['C2H6']+df['CO']+df['CO2'])/18 # from An approach to power transformer asset management using health index
    df["DGAF"] = df["H2"]
    df = DGAF_score(df)
    df.drop(['H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2'],axis=1,inplace=True)
    df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    df = df.dropna()
    
    # Get PTs with more than k(=25) records
    x = df["SAPID"].value_counts()
    x = x[x>25] # arbitrary number
    x = x.index

    df = df[df["SAPID"].isin(x)]

    #gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
    #gpr.fit(np.array(df["Data Colheita"]).reshape(-1,1), np.array(df["DGAF"]).reshape(-1,1))
    
    #return gpr.predict(np.atleast_2d(np.linspace(0,1000,1000)).T)
    # Tried bagging, having trouble with the last part, element wise multiplication of np arrays


    # Separating df by PT
    grouped = df.groupby(df["SAPID"])
    models  = []
    print("Number of Gps:",len(grouped))
    counter = 0
    #return grouped
    for PT in grouped:
        counter+=1
        print("Currently on GP number",counter,"/",len(grouped))
        PT = pd.DataFrame(list(PT)[1]) # Convoluted way of converting to data frame
        PT["Data Colheita"] = (PT["Data Colheita"] - PT.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age

        '''
        import matplotlib.pyplot as plt
        axes = plt.gca()
        axes.set_ylim([-10,50])
        plt.scatter(PT["Data Colheita"],PT["DGAF"],s=10)
        #plt.ylim(3000)
        plt.show()
        return
        '''
        PT.reset_index(drop=True, inplace=True)
        current_features = []
        current_label = []
        cur_start = 0 # to effectively start counting from scratch when oil change
        
        for index, row in PT.iterrows(): # splitting when oil change
            if index == 0:
                current_features.append(0)
                #current_label = [1]
                current_label = [0]
                continue
            if PT.iloc[index,2] >= 0.6*PT.iloc[index-1,2]: # assuming this implies oil change
                
                current_features.append(row["Data Colheita"] - cur_start)
                #current_label.append(row["DGAF"])
                #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
            else:
                gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
                if not current_features:
                    current_features.append(row["Data Colheita"] - cur_start)
                    #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                    current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
                current_features = np.array(current_features).reshape(-1,1)
                current_label = np.array(current_label).reshape(-1,1)
                gpr.fit(current_features, current_label)
                models.append(gpr)
                cur_start = row["Data Colheita"]
                current_features = [row["Data Colheita"] - cur_start]
                #current_label = [1]
                current_label = [0]
        if current_features:#len(current_features) > 10:
            current_features = np.array(current_features).reshape(-1,1)
            current_features = np.array(current_label).reshape(-1,1)
            gpr.fit(current_features, current_label)
    
    y_hat   = []
    weights = []
    res = []
    X_test = np.atleast_2d(np.linspace(0,6000,1000)).T
    for model in models:
            prediction, std = model.predict(X_test, return_std=True)
            y_hat.append(prediction.reshape(1,-1))
            weights.append(std)

    weights = 1/np.array(weights)
    weight_sum = np.array(weights).sum(axis=0)
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j]/=weight_sum[j]
    

    for i in range(len(y_hat)):
        y_hat[i] = np.ravel(y_hat[i])

    model_prediction = np.multiply(y_hat, weights) # hadamard product
    np.set_printoptions(suppress=True)

    return np.sum(model_prediction,axis=0)
    

def predictDGA(kernel):
    _, df = really()
    df["DGAF"] = (2*df['H2']+3*df['CH4']+5*df['C2H2']+3*df['C2H4']+3*df['C2H6']+df['CO']+df['CO2'])/18 # from An approach to power transformer asset management using health index
    df.drop(['H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2'],axis=1,inplace=True)
    df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    df = df.dropna()
    
    # Get PTs with more than k(=25) records
    x = df["SAPID"].value_counts()
    x = x[x>35] # arbitrary number
    x = x.index


    df = df[df["SAPID"].isin(x)]
    grouped = df.groupby(df["SAPID"])
    print("Number of PTs:",len(grouped))

    gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
    counter = 0
    for PT in grouped:
        counter+=1
        print("Currently on PT number",counter,"/",len(grouped))
        PT = pd.DataFrame(list(PT)[1]) # Convoluted way of converting to data frame
        PT["Data Colheita"] = (PT["Data Colheita"] - PT.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
        PT.reset_index(drop=True, inplace=True)
        current_features = []
        current_label = []
        cur_start = 0 # to effectively start counting from scratch when oil change
        
        for index, row in PT.iterrows(): # splitting when oil change
            #print(row["Data Colheita"])
            #print(row["DGAF"])
            #print()
            if index == 0:
                current_features.append(0)
                #current_label = [1]
                current_label = [0]
                continue
            if PT.iloc[index,2] >= 0.6*PT.iloc[index-1,2]: # assuming this implies oil change
                
                current_features.append(row["Data Colheita"] - cur_start)
                #current_label.append(row["DGAF"])
                #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
            else:
                if not current_features:
                    current_features.append(row["Data Colheita"] - cur_start)
                    #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                    current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
                current_features = np.array(current_features).reshape(-1,1)
                current_label = np.array(current_label).reshape(-1,1)
                print(current_features)
                print(current_label)
                gpr.fit(current_features, current_label)
                cur_start = row["Data Colheita"]
                current_features = [row["Data Colheita"] - cur_start]
                current_label = [0]
        return PT

        if current_features:#len(current_features) > 10:
            current_features = np.array(current_features).reshape(-1,1)
            current_features = np.array(current_label).reshape(-1,1)
            gpr.fit(current_features, current_label)

    np.set_printoptions(suppress=True)
    X_test = np.atleast_2d(np.linspace(0,6000,1000)).T
    prediction, std = gpr.predict(X_test, return_std=True)
    return prediction
    #output = {:.9f}.format(prediction)
    return output



def trying_linear_regression():
    _, df = really()
    #df["DGAF"] = (2*df['H2']+3*df['CH4']+5*df['C2H2']+3*df['C2H4']+3*df['C2H6']+df['CO']+df['CO2'])/18 # from An approach to power transformer asset management using health index
    df["DGAF"] = df["H2"]
    df = DGAF_score(df)
    df.drop(['H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2'],axis=1,inplace=True)
    df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    df = df.dropna()
    
    # Get PTs with more than k(=25) records
    x = df["SAPID"].value_counts()
    x = x[x>25] # arbitrary number
    x = x.index


    df = df[df["SAPID"].isin(x)]
    grouped = df.groupby(df["SAPID"])
    print("Number of PTs:",len(grouped))

    reg = LinearRegression()
    global_features = []
    global_label = []
    models = []
    counter = 0
    for PT in grouped:
        counter+=1
        print("Currently on PT number",counter,"/",len(grouped))
        PT = pd.DataFrame(list(PT)[1]) # Convoluted way of converting to data frame
        PT["Data Colheita"] = (PT["Data Colheita"] - PT.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
        PT.reset_index(drop=True, inplace=True)
        current_features = []
        current_label = []
        cur_start = 0 # to effectively start counting from scratch when oil change
        for index, row in PT.iterrows(): # splitting when oil change
            if index == 0:
                current_features.append(0)
                #current_label = [1]
                current_label = [0]
                continue
            if PT.iloc[index,2] >= 0.6*PT.iloc[index-1,2]: # assuming this implies oil change
                
                current_features.append(row["Data Colheita"] - cur_start)
                #current_label.append(row["DGAF"])
                #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
            else:
                if not current_features:
                    current_features.append(row["Data Colheita"] - cur_start)
                    #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                    current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
                reg = LinearRegression()
                current_features = np.array(current_features).reshape(-1,1)
                current_label = np.array(current_label).reshape(-1,1)
                reg.fit(current_features, current_label)
                models.append(reg)
                cur_start = row["Data Colheita"]
                global_features.extend(current_features)
                global_label.extend(current_label)
                current_features = [row["Data Colheita"] - cur_start]
                current_label = [0]
        

        if current_features:#len(current_features) > 10:
            current_features = np.array(current_features).reshape(-1,1)
            current_features = np.array(current_label).reshape(-1,1)
            reg = LinearRegression()
            reg.fit(current_features, current_label)
            models.append(reg)
    import matplotlib.pyplot as plt
    plt.scatter(global_features, global_label,s=10)
    plt.ylim([-1,2000])
    plt.show()

    np.set_printoptions(suppress=True)

    y_hat   = []
    res = []
    X_test = np.atleast_2d(np.linspace(0,6000,1000)).T
    return [global_features, global_label]
    return models
    for model in models:
            prediction = model.predict(X_test)
            y_hat.append(prediction.reshape(1,-1))
    return y_hat
    #for i in range(len(y_hat)):
    #    y_hat[i] = np.ravel(y_hat[i])

    model_prediction = np.mean(y_hat)

    return model_prediction


def trying_linear_regression(df):
    #df["DGAF"] = (2*df['H2']+3*df['CH4']+5*df['C2H2']+3*df['C2H4']+3*df['C2H6']+df['CO']+df['CO2'])/18 # from An approach to power transformer asset management using health index
    df["DGAF"] = df["H2"]
    df = DGAF_score(df)
    
    df.drop(['H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2'],axis=1,inplace=True)
    df["Data Colheita"] = df['Data Colheita'].astype('datetime64[ns]')
    #df["DGAF"] = 100 - (df["DGAF"]-1)*100/5 <- convert it to 0-100
    df = df.dropna()
    
    # Get PTs with more than k(=25) records
    x = df["SAPID"].value_counts()
    x = x[x>25] # arbitrary number
    x = x.index
    

    df = df[df["SAPID"].isin(x)]
    grouped = df.groupby(df["SAPID"])
    print("Number of PTs:",len(grouped))
    
    reg = LinearRegression()
    global_features = []
    global_label = []
    models = []
    counter = 0
    reg = LinearRegression()
    
    for PT in grouped:
        counter+=1
        print("Currently on PT number",counter,"/",len(grouped))
        PT = pd.DataFrame(list(PT)[1]) # Convoluted way of converting to data frame
        PT["Data Colheita"] = (PT["Data Colheita"] - PT.iloc[0]["Data Colheita"]).dt.days # Convoluted way of converting dates to age
        PT.reset_index(drop=True, inplace=True)
        current_features = []
        current_label = []
        cur_start = 0 # to effectively start counting from scratch when oil change
        for index, row in PT.iterrows(): # splitting when oil change
            if index == 0:
                #current_features.append(0)
                #current_label = [1]
                current_features.append(row["DGAF"])
                current_label = [0]
                continue
            if PT.iloc[index,2] >= 0.6*PT.iloc[index-1,2]: # assuming this implies oil change
                current_features.append(row["Data Colheita"] - cur_start)
                #current_label.append(row["DGAF"])
                #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                #current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
                current_label.append(PT.iloc[index,2])
            else:
                if not current_features:                
                    current_features.append(row["Data Colheita"] - cur_start)
                    #current_label.append(PT.iloc[index,2]/PT.iloc[index-1,2])
                    #current_label.append(PT.iloc[index,2]-PT.iloc[index-1,2])
                    current_label.append(PT.iloc[index,2])
                cur_start = row["Data Colheita"]
                global_features.extend(current_features)
                global_label.extend(current_label)
                current_features = [row["Data Colheita"] - cur_start]
                current_label = [0]
    

    import matplotlib.pyplot as plt
    reg.fit(np.array(global_features)[:,np.newaxis], np.array(global_label)[:,np.newaxis])
    return reg
    plt.scatter(global_features, global_label,s=10)
    plt.ylim([0.5,6.5])
    X_test = np.atleast_2d(np.linspace(0,9000,1500)).T
    plt.xlabel('Days since oil change')
    plt.ylabel('DGA Score')
    plt.plot(X_test, reg.predict(X_test),color='red')
    plt.show()

    np.set_printoptions(suppress=True)
    return global_label
    return df

    return model_prediction


# Towards a comprehensive health index
def DGAF_score(df):
    df["DGAF"] = df["H2"]
    Htwo = [100,200,300,500,700]
    CHfour = [75,125,200,400,600]
    CtwoHsix = [65,80,100,120,150]
    CtwoHfour = [50,80,100,150,200]
    CtwoHtwo = [3,7,35,50,80]
    CO = [350,700,900,1100,1400]
    COtwo = [2500,3000,4000,5000,7000]
    dga_gases = ['H2','CH4','C2H6','C2H4','C2H2','CO','CO2']
    
    scores = [Htwo, CHfour, CtwoHsix, CtwoHfour, CtwoHtwo, CO, COtwo]
    weights = [2,3,3,3,5,1,1]
    for i, row in df.iterrows():
        result = 0
        for index, gas in enumerate(dga_gases):
            counter = 1
            for j in scores[index]:
                if row[gas] <= j:
                    break
                counter+=1
            result+=(counter*weights[index])#*row[gas]
        #df.iat[i,df.columns.get_loc("DGAF")] = result/sum(weights)
        df.at[i,"DGAF"] = result/sum(weights)

    
    return df


#def rate_of_change_score


if __name__ == "__main__":
    kernel = RationalQuadratic(length_scale_bounds=(0.1, 200)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1))
    #x = predictDGA_bagging(kernel)
    _, df = really()
    x = trying_linear_regression(df)
    #print(x)
    #x = predictDGA(kernel)
    #print(DGAF_score(predictDGA_bagging(kernel)))
