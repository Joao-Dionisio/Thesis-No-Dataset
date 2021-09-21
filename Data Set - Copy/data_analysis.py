import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from read import read_file
import numpy as np
#import seaborn as sns
#from statistics import mean
#from pandas.api.types import is_numeric_dtype
#from pandas_profiling import ProfileReport
#from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn import preprocessing

#dga              = ['H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2']
#info             = ['SAPID', 'Instalacao', 'Transformador', 'N_Serie_LABELEC', 'Data Colheita']
#testes_eletricos = ['Tensao_Interfacial', 'Tensao_Disruptiva', 'Tangente_Delt_90']
#outros_testes    = ['2FAL', 'COR']
#diagnosticos     = ['Diagnostico_furanicos', 'Diagnostico_AC', 'Diagnostico_FQ', 'Recomendacoes']


def really():
    df = pd.read_csv("Dados_PATH.csv")
    df = df.apply(lambda x: x.astype(str).str.replace(',','.'))
    df["Data Colheita"] = pd.to_datetime(df["Data Colheita"])

    df[["SAPID","2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2']] = df[["SAPID","2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2']].apply(pd.to_numeric, errors='coerce')
    df = df[df[["SAPID", "2FAL", 'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2', "Data Colheita"]].notna()]
    
    twoFAL = df[["SAPID", "2FAL", "Data Colheita"]]

    DGA = df[["SAPID",'H2','CH4','C2H2','C2H4','C2H6','CO','CO2','O2','N2', 'Data Colheita']]
    return twoFAL, DGA

a = really()

def data_exploration():
    df = pd.read_csv("Dados_PATH.csv")
    #df = pd.read_csv("RARI2020.csv")
    #df.columns = [c.replace(' ', '_') for c in df.columns]
    #profile = ProfileReport(df, title = "Pandas Profiling Report")
    #profile.to_file("your_report.html")
    df = df.apply(lambda x: x.astype(str).str.replace(',','.'))
    #df = df.apply(lambda x: x.astype(str).str.replace('novo','-9999999'))
    #df = df.apply(lambda x: x.astype(str).str.replace('nan','0'))
    
    df["Data Colheita"] = pd.to_datetime(df["Data Colheita"])

    df["SAPID"] = pd.to_numeric(df["SAPID"], errors="coerce")

    # Working with 2FAL only
    df = df[df['2FAL'].notna()]
    df = df[df['SAPID'].notna()]
    df = df[df['Data Colheita'].notna()]
    df = df[["SAPID", "2FAL", "Data Colheita"]]

    # Working with DGA only
    

    
    #time_series(df, "Idade (em 2015)", "2FAL final")
    df2 = pd.read_csv("TP ATMT_Avarias (actualização 19Junho2018).csv") # Failure dataset
    df3 = read_file('RARI2020.xlsx', 3)
    df3["SAP ID"] = df3["SAP ID"].astype(object)
    df3.rename({"SAP ID":"SAPID", "Idade \n(em 2015)":"Idade (em 2015)"}, axis = 1, inplace=True)
    df3[["2FAL final", "Idade (em 2015)"]] = df3[["2FAL final", "Idade (em 2015)"]].apply(pd.to_numeric)
    #df3[["H2","C2H2","C2H4", "C2H6", "CO2", "CO", "2 FAL 2019", "Teor de Acidez", "Teor de Água [ppm]"]] = df[["H2","C2H2","C2H4", "C2H6", "CO2", "CO", "2 FAL 2019", "Teor de Acidez", "Teor de Água [ppm]"]].apply(pd.to_numeric)

    df2.rename({"SAP ID":"SAPID"}, axis = 1, inplace=True)
     
    df["SAPID"] = pd.to_numeric(df["SAPID"], errors="coerce")
    df2["SAPID"] = pd.to_numeric(df2["SAPID"], errors="coerce")
    df3["SAPID"] = pd.to_numeric(df3["SAPID"], errors="coerce")

    return df2, df

    df = pd.merge(df, df2[["SAPID","Idade"]], on="SAPID", how="left")
    
    df = pd.merge(df, df3[["SAPID","Ano fabrico"]], on="SAPID", how="left")
    df = df[df['2FAL'].notna()]
    df = df[df['SAPID'].notna()]
    
    df["Idade na Recolha"] = pd.DatetimeIndex(df['Data Colheita']).year - df["Ano fabrico"]
    
    #df = df[df["Idade na Recolha"] >= 0]
    df = df[df["SAPID"] != 280898222] # These are the only transformers that have negative Idade na Recolha. I asusme that Ano fabrico is wrong. 
    df = df[df["SAPID"] != 280192286]
    df = df[df['Idade na Recolha'].notna()]    
    
    
    df[["2FAL"]] = df[["2FAL"]].apply(pd.to_numeric, errors = "coerce")
    df = df[df['2FAL'].notna()]    
    
    
    #maintenance_according_to_factor(df, "Idade na Recolha", "2FAL")

    return feature_importance(df, "Idade")

    return df, df3 
    
    pca(df) # way too early for this
    #for i in dga:
    #    mean_from_category(df, 'Diagnostico_AC', i)
    #stacked_DGA(df)
    return

    df['CO2/CO'] = df['CO2']/df['CO']
    
    return
    dga_var = df.iloc[8:16]
    get_heatmap(dga_var)
    return
    separated_df = separate_by_factor(df, 'Diagnostico_AC')
    for i in separated_df:
        get_heatmap(separated_df[i])

def normalize(df):
    return (df-df.mean())/df.std()

def data_cleaning(df):
    df = df.apply(lambda x: x.astype(str).str.replace(',','.'))
    df = df.apply(lambda x: x.astype(str).str.replace('novo','-9999'))
    df = df.apply(lambda x: x.astype(str).str.replace('nan','-9999'))

    print(list(df.columns))
    #df[["Ano fabrico", "H2","C2H2","C2H4", "C2H6", "CO2", "CO", "Tensão Disruptiva [kV]","2 FAL 2019", "Teor de Acidez", "Teor de Água [ppm]", "Cor"]] = df[["Ano fabrico", "H2","C2H2","C2H4", "C2H6", "CO2", "CO", "Tensão Disruptiva [kV]", "2 FAL 2019", "Teor de Acidez", "Teor de Água [ppm]", "Cor"]].apply(pd.to_numeric)
    #df[["H2","C2H2","C2H4", "C2H6", "CO2", "CO", "Tensão Disruptiva [kV]", "Teor de Acidez", "Teor de Água [ppm]", "Cor", "IFT [dyne/cm]", "2 FAL 2019", "tan d", "LF", "PQ C", "PQ B", "PQ A", "PQ A+", "Total PQ", "Aceleração Sísmica solo (m/s2)", "Transformador %", "TIEPI"]] = df[["H2","C2H2","C2H4", "C2H6", "CO2", "CO", "Tensão Disruptiva [kV]", "2 FAL 2019", "Teor de Acidez", "Teor de Água [ppm]", "Cor", "IFT [dyne/cm]", "tan d", "LF", "PQ C", "PQ B", "PQ A", "PQ A+", "Total PQ", "Aceleração Sísmica solo (m/s2)", "Transformador %", "TIEPI"]].apply(pd.to_numeric)
    df = df.apply(pd.to_numeric, errors='ignore')


    #df = normalize(df, ["H2","C2H2","C2H4", "C2H6", "CO2", "CO", "2FAL", "Massa_Volumica", "Tensao_Interfacial", "Tensao_Disruptiva", "Indice_Acidez", "Teor_Agua", "Tangente_Delta_90"])
    # For some reason, this ^ makes all the records with the same SAP ID equal.
    
    #df['Idade'] = 2021-df['Ano fabrico']
    #print(df['Idade'].unique())
    df = df[pd.to_numeric(df['SAP ID'], errors='coerce').notnull()]
    df["DP"] = (1.51 - np.log10(df["2FAL"]))/0.0035
    
    return df


def feature_importance(df, target):
    #df = read_file('RARI2020.xlsx',3)
    #df = data_cleaning(df)

    
    #df[target] = df[target].fillna(0)
    df = df[df[target].notna()]
    
    rec = df[[target]]

    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

    df.drop(["Idade", "Data Colheita"], axis=1, inplace=True)
    forest.fit(df, rec)
        
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print('bbbb')
    print("Feature ranking:")

    for f in range(df.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, df.columns[indices[f]], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(df.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(df.shape[1]), indices)
    plt.xlim([-1, df.shape[1]])
    plt.show()
    return df

def feature_importance2(df, target):
    df = data_cleaning(df)

    #df = df[df[target].notna()]
    df['Ano Avaria'] = df['Ano Avaria'].fillna(0, inplace=True)
    rec = df[[target]]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    df['CO2/CO'] = df['CO']/df['CO2']

    clf = LogisticRegression().fit(df, rec)
    result = permutation_importance(clf, df, rec, n_repeats=10, random_state=0)

    print(result.importances_mean)
    print(result.importances_std)
    return df
    

def time_series(df, label, factor):
    #from itertools import cycle
    #colors = cycle(["aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "yellow"])

    seperated_df = separate_by_factor(df, 'SAPID') 
    index = 0
    for cur in seperated_df:    
        if not seperated_df[cur].empty and len(seperated_df[cur][factor]) > 20 and index <= 2:
            x = seperated_df[cur][label]
            y = seperated_df[cur][factor]
            #if is_numeric_dtype(y):
            #    y/=max(y)
            #color = next(colors)
            plt.plot(x, y)#, color)#, kind = 'line')
            index+=1
    
    #seperated_df[cur][factor]/max(seperated_df[cur][factor])
    print(index)
    plt.xlabel("Data da Colheita")
    plt.ylabel('2FAL')
    plt.show()

    '''
    from matplotlib import pyplot as plt
    plt.plot(df["Data Colheita"], df["DP"])
    plt.show()
    '''



def separate_by_factor(df, factor):

    # Separating the readings by given factor
    UniqueFactor = df[factor].unique()

    #create a data frame dictionary to store your data frames
    DataFrameDict = {elem : pd.DataFrame for elem in UniqueFactor}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = df[:][df[factor] == key]
    
    return DataFrameDict

def maintenance_according_to_factor(df, label, value):
    print(df[[value, label]].groupby(label).mean())
    

    temp = df[df["Idade"].notna()]
    plt.bar(temp["Idade"], temp["Idade na Recolha"]/(3*max(df[value])), color = "red")
    plt.plot(df[[value, label]].groupby(label).mean())

    plt.xlabel('Idade Transformador')
    plt.ylabel('2FAL standardized')
    
    plt.show()


############################# Outras coisas que estava a fazer #################################


def mean_from_category(df, label, value):
    separated_df = separate_by_factor(df, label)
    temp = df.loc[df[label].notna()]
    #sns.catplot(x = 'Diagnostico_furanicos', y='2FAL', kind = 'bar', data=temp)
    #plt.show()

    _val = []
    _label = []
    for i in separated_df:
        if not separated_df[i].empty:
            _label.append(i)
            temp = mean(separated_df[i][value])
            if np.isnan(temp):
                _val.append(temp)
    y = sorted(_val)
    x = [_label[_val.index(i)] for i in y]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x, y)  # 2FAL increases as the paper degradation increases
    rects = ax.patches
    unnecessary = [' sintomas', 'de ','do ', 'papel ','Papel ', ' óleo', '.','Sintomas ', 'papel', ' isolante', ' da']
    labels = x
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        for i in unnecessary:
            label = label.replace(i, '')
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.1, label,
                ha='center', va='bottom')
    plt.show()



def stacked_DGA(df):


    separated_df = separate_by_factor(df, 'Diagnostico_AC')
    DGA = []
    for i in range(len(separated_df)):
        DGA.append([0 for i in range(len(dga))])

    for index, label in enumerate(separated_df):
        if not separated_df[label].empty:
            for gas in dga:
                DGA[index][dga.index(gas)] = mean(separated_df[label][gas])

    H2   = [DGA[index][0] for index in range(len(separated_df))]
    CH4  = [DGA[index][1] for index in range(len(separated_df))]
    C2H2 = [DGA[index][2] for index in range(len(separated_df))]
    C2H4 = [DGA[index][3] for index in range(len(separated_df))]
    C2H6 = [DGA[index][4] for index in range(len(separated_df))]
    CO   = [DGA[index][5] for index in range(len(separated_df))]
    CO2  = [DGA[index][6] for index in range(len(separated_df))]
    O2   = [DGA[index][7] for index in range(len(separated_df))]
    N2   = [DGA[index][8] for index in range(len(separated_df))]

    labels = [i for i in separated_df]
    labels[1] = 'Error' # 1 is nan, for some reason
    for gas in [H2,CH4,C2H2,C2H4,C2H6,CO,CO2,O2,N2]:
        gas = [i/max(gas) for i in gas]
        plt.bar(labels, gas)
    

    plt.show()



def pca(df):
    from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=1)
    df1 = df[['Massa_Volumica','Tensao_Interfacial','Tensao_Disruptiva','Indice_Acidez','Teor_Agua','Tangente_Delta_90']]
    df1 = StandardScaler().fit_transform(df1)
    df2 = pd.DataFrame(data=df1,  
              index=df1[0:,0],  
              columns=df1[0,0:])
    df2['Diagnostico_FQ'] = df['Diagnostico_FQ']
    
    df2 = df2.dropna()
    principalComponents = pca.fit(df2.loc[:, df2.columns != 'Diagnostico_FQ'])
    df1['pca'] = principalComponents
    separated_df = separate_by_factor(df1, 'Diagnostico_FQ')
    
    _val = []
    _label = []
    for i in separated_df:
        if not separated_df[i].empty:
            _label.append(i)
            temp = mean(separated_df[i]['pca'])
            if not np.isnan(temp):
                _val.append(temp)
    y = sorted(_val)
    x = [_label[_val.index(i)] for i in y]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x, y)
    plt.show()
    return principalComponents


if __name__ == "__main__":
    #df, df3 = data_exploration()
    df = data_exploration()
    #df = feature_importance2()
