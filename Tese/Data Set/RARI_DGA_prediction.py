# Introducing Load Factor into DGA prediction
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from predict_DGA import DGAF_score, trying_linear_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def xlsx_to_csv():
    data_xls = pd.read_excel('RARI2020.xlsx', skiprows=3,skipfooter=790,index_col=None)
    data_xls.to_csv('RARI2020.csv', encoding='utf-8', index=False)

def lf_separation(model, entire_df):
    
    df = pd.read_csv("RARI2020.csv")
    df.columns = [c.replace(' ', '_') for c in df.columns]

    df = df[["SAP_ID", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','LF', "Idade"]]
    df[["SAP_ID", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','LF', "Idade"]] = df[["SAP_ID", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','LF', "Idade"]].apply(pd.to_numeric, errors='coerce') 
    df['O2'], df['N2'] = df['H2'], df['H2'] # Rari as fewer gases than PATH
    #df = df.apply(lambda x: x.astype(str).str.replace(',','.'))

    #df[["SAPID","2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2']] = df[["SAPID","2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2']].apply(pd.to_numeric, errors='coerce')

    df['Idade']=df['Idade'].astype(float)
    df['LF']=df['LF'].astype(float)
    df['LF']=df['LF']*100
    
    # Working with DGA
    df = df.dropna()
    df = df[df.notna()]
    df = DGAF_score(df)
    if model == 'logistic':
        df['DGAF'] = np.where(df['DGAF']<1.5,0,1) 
    
    if entire_df:
        return df
    
    df1 = df.loc[(0 <= df["LF"]) & (df["LF"] < 10)]
    df2 = df.loc[(10 <= df["LF"]) & (df["LF"] < 20)]
    df3 = df.loc[(20 <= df["LF"]) & (df["LF"] < 30)]
    df4 = df.loc[(40 <= df["LF"]) & (df["LF"] < 40)]
    df5 = df.loc[(40 <= df["LF"]) & (df["LF"] < 50)]
    df6 = df.loc[(50 <= df["LF"]) & (df["LF"] < 60)]
    df7 = df.loc[(60 <= df["LF"]) & (df["LF"] < 70)]
    df8 = df.loc[(70 <= df["LF"]) & (df["LF"] < 80)]
    df9 = df.loc[(80 <= df["LF"]) & (df["LF"] < 90)]
    df10 = df.loc[(90 <= df["LF"]) & (df["LF"] <= 100)]
    df11 = df.loc[(100 < df["LF"])]
    return [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11]

    


def threed_linear_regression():
    df = lf_separation(True)
    #df["DGAF"] = (2*df['H2']+3*df['CH4']+5*df['C2H2']+3*df['C2H4']+3*df['C2H6']+df['CO']+df['CO2'])/18 # from An approach to power transformer asset management using health index
    df["DGAF"] = df["H2"]
    df = DGAF_score(df)
    
    df.drop(['H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2'],axis=1,inplace=True)
    
    grouped = df.groupby(df["SAP_ID"])

    X = df[["Idade","LF"]]
    y = df["DGAF"].values.reshape(len(df["DGAF"]),1)
    
    reg = LinearRegression()
    reg.fit(X, y)

    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.scatter(X['Idade'],X['LF'], y,s=10)
    
    x_pred = np.linspace(0,max(X['Idade']),100)
    y_pred = np.linspace(0,100,100)
    xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)

    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()])
    
    ax.set_xlabel('Years since oil change')
    ax.set_ylabel('LF')
    ax.set_zlabel('DGA Score')
    
    ax.scatter(model_viz[0], model_viz[1], reg.predict(model_viz.T),color='wheat', alpha=0.2, depthshade=0)
    
    #ax.scatter(model_viz[0], model_viz[1], reg.predict(model_viz),color='red')
    plt.show()

    np.set_printoptions(suppress=True)
    return 
    return global_label
    return df

    return model_prediction




def linear_regression(df,model):
    
    #df["DGAF"] = df["H2"]
    #df = DGAF_score(df)
    
    #df.drop(['H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2'],axis=1,inplace=True)
    #df["DGAF"] = df["DGAF"].astype(int) #<- Logistic Regression needs categorical data 
    
    X = df["Idade"].values.reshape(-1,1)
    y = df["DGAF"].values.reshape(-1,1)

    if model == 'logistic':
        reg = LogisticRegression()
    else:
        reg = LinearRegression()

    reg.fit(X, y)
    
    plt.scatter(X,y,s=10)
    
    X_test = np.linspace(0,100,10000)
    print('Slope:',round(reg.coef_[0][0],4))
    print(reg.score(X,y))
    plt.plot(X_test, reg.predict(X_test.reshape(-1,1)))
    
    #ax.scatter(model_viz[0], model_viz[1], reg.predict(model_viz),color='red')
    plt.show(block=False)
    plt.pause(0.5)
    
    np.set_printoptions(suppress=True)
    return 
    return global_label
    return df

    return model_prediction




def main(df):
    clf = LogisticRegression()
    X = df["Idade"].values.reshape(-1,1)
    y = df[""]
    clf.fit(df["Idade"],df["DGAF"])

    X_test = np.linspace(0,100,1000)
    
    

if __name__ == "__main__":
    #xlsx_to_csv()
    #x = lf_separation()

    #threed_linear_regression()
    model = 'logistic'
    x = lf_separation(model, entire_df=False)

    counter = 0
    
    for i in x:
        counter+=1
        x = i
        if len(i) > 10 and (i['DGAF'] != i['DGAF'].iloc[0]).any():
            if counter == 11:
                print('Current load higher than 100%')
            else:
                print('Current load lower than', str(10*counter)+'%')
            x=linear_regression(i,model)
            #linear_regression(pd.DataFrame(list(i)[1]))
