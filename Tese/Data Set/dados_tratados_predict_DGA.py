import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier

#data_xls = pd.read_excel('dados_tratados.xlsx', dtype=str, index_col=None)
#data_xls.to_csv('dados_tratados.csv', encoding='utf-8', index=False)


def load_dados_tratados(df):
    '''
    Loads dados_tratados.csv dataset and introduces the oil RUL variable

    '''
    df = pd.read_csv('dados_tratados.csv')
    df = df[df.notna()]
    #df = df[["Transformador_ID","DataColheita","OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","cond(percentagem)","fal"]]
    #df = df[["Transformador_ID","DataColheita","OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","fal"]]
    df['TIPO TP'] = df['TIPO TP'].map({'A': 0, 'B': 1, 'O':2})
    
    #df = df[["quantas_trocas_oleo", "TIPO TP", "Idade", "TemperaturaOleo", "CO2", "O2", "CO", "X2ACF", "X2FOL", "X5HMF", "X5MEF", "OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","fal"]]
    #df = df[["Transformador_ID","OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","fal"]]
    df["DataColheita"] = df['DataColheita'].astype('datetime64[ns]')
    
    df['Oil_RUL'] = df.groupby(['Transformador_ID','quantas_trocas_oleo'])['DataColheita'].apply(lambda x: (x.max() - x))
    
    #df['Oil_RUL'] = df['Oil_RUL']//365

    df.drop(['Grupo ID','AnoFabrico','ReferenciaLabelec','Local','AnoFabrico','Tipo_ref','cond(percentagem)','cond_oleo'],axis=1,inplace=True)

    df['Oil_RUL'] = round(df["Oil_RUL"].dt.days, 1)
    #df['fal'] = pd.cut(df.fal, bins=[0,0.1,0.25,0.5,1,np.inf], labels=[1,2,3,4,5])

    return df

    '''
    #df[["Transformador_ID", "OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","cond(percentagem)"]] = df[["Transformador_ID", "OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","cond(percentagem)"]].apply(pd.to_numeric, errors='coerce')
    df[["Transformador_ID", "OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","fal"]] = df[["Transformador_ID", "OVER_horas","Q1_horas","Q2_horas","Q3_horas","Q4_horas","fal"]].apply(pd.to_numeric, errors='coerce')
    
    df["cond(percentagem)"] = 100*df["cond(percentagem)"]
    #df["cond(percentagem)"] = df["cond(percentagem)"].astype(int)
    df["cond(percentagem)"] = np.where(df["cond(percentagem)"]<25,0,1)

    #return df["cond(percentagem)"]

    df = df.sort_values(['Transformador_ID', 'DataColheita'], ascending = [True,True])
    df = df.round(2)
    '''
    
    grouped = df.groupby(df["Transformador_ID"])
    
    print("Number of PTs:",len(grouped))
    #model = LinearRegression() # <- linear regression is not good, R^2 is 0.03
    model = LogisticRegression()
    
    counter = 0
    
    X_train = []
    y_train = []
    for PT in grouped:
        counter+=1
        print("Currently on PT number",counter,"/",len(grouped))
        PT = pd.DataFrame(list(PT)[1]) # Convoluted way of converting to data frame
        PT["DataColheita"] = (PT["DataColheita"] - PT.iloc[0]["DataColheita"]).dt.days # Convoluted way of converting dates to age
        PT.reset_index(drop=True, inplace=True)
        current_features = []
        current_label = []
        cur_start = 0 # to effectively start counting from scratch when oil change
        
        for index, row in PT.iterrows(): # splitting when oil change
            if index == 0:
                continue
            #current_features = [row["DataColheita"] - PT.iloc[index-1]["DataColheita"], row["OVER_horas"],row["Q1_horas"],row["Q2_horas"],row["Q3_horas"],row["Q4_horas"]]
            current_features = [row["OVER_horas"],row["Q1_horas"],row["Q2_horas"],row["Q3_horas"],row["Q4_horas"]]          
            current_label = [row["cond(percentagem)"]]
            X_train.append(current_features)
            y_train.append(current_label)
        
    model.fit(X_train, np.array(y_train).ravel())

    print(model.score(X_train, y_train))

    np.set_printoptions(suppress=True)
    #print(reg.coef_)
    #print(reg.intercept_)
    #print(reg.score(X_train,y_train))
    
    return [X_train, y_train]
    print(y_train)


def plot_censoring_data(df):
    '''
    Plot oil runs and labels depending on right censoring and
    left truncation
    '''
    
    grouped_oil = df.groupby(['Transformador_ID', 'quantas_trocas_oleo'])
    grouped_PTs = df.groupby(['Transformador_ID'])
    right_endpoint = []
    left_endpoint = []
    right_censoring = []
    
    for PT in grouped_oil:
        PT = pd.DataFrame(list(PT)[1])
        right_endpoint.append(PT['Idade'].max())
        left_endpoint.append(PT['Idade'].min())
        right_censoring.append(df[df['Transformador_ID'] == PT['Transformador_ID'].iloc[0]]['Idade'].max() == PT['Idade'].max())

    for i,j in enumerate(right_endpoint):
        if left_endpoint[i] > 0 and right_censoring[i]:
            color = 'red'
        elif left_endpoint[i] > 0:
            color = 'yellow'
        elif right_censoring[i]:
            color = 'blue'
        else:
            color = 'green'
        plt.plot([left_endpoint[i],j],[3*i,3*i],'ro-', lw=0.3, markersize=1, markeredgecolor=color, color=color)     
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Left Truncation + Right Censoring', markerfacecolor='red', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Left Truncation', markerfacecolor='yellow', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Right Censoring', markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='All good', markerfacecolor='green', markersize=15)]
    plt.legend(handles=legend_elements, prop={'size': 10}, markerscale=0.6)
    #for handle in lgnd.legendHandles:
    #    handle.set_sizes([6.0])
    #plt.show()
    return df


def kaplan_meier(df):
    '''
    Uses Kaplan-Meier estimator in order to combact right censored data.
    It does not work, there is simply too much censored data
    '''
    df['Event'] = 0
    
    max_age = df['Idade'].max()
    grouped_oil = df.groupby(['Transformador_ID', 'quantas_trocas_oleo'])
    #grouped_pts = df.groupby(['Transformador_ID'])
    #return [grouped_oil, grouped_pts]
    death_time = max_age*[0]
    susceptible = max_age*[0] #susceptible oil at time t (-1)
    censored = max_age*[0]
    
    #return [grouped_pts, grouped_oil]
    for oil_run in grouped_oil:
        oil_run = pd.DataFrame(list(oil_run)[1])

        # If max oil age < max PT age, then oil 'died', otherwise it was censored 
        if oil_run['Idade'].max() < df[df['Transformador_ID'] == oil_run['Transformador_ID'].iloc[0]]['Idade'].max():
            death_time[oil_run['Idade'].max()-1]+=1
            df.iat[oil_run['Idade'].idxmax(), df.columns.get_loc('Event')] = 1
        else:
            censored[oil_run['Idade'].max()-1]+=1
    

        
    ages = df['Idade'].value_counts().index.tolist() # Number of PT ages
    occurences = list(df['Idade'].value_counts()) # number of PTs with each age

    for i in range(len(occurences)):
        susceptible[ages[i]-1] = occurences[i]

    survival_function = [1]
    censor_function = [1]
    p0 = 1
    p1 = 1
    for i in range(1,df['Idade'].max()):
        if susceptible[i]:
            sus_p = (susceptible[i]- death_time[i])/susceptible[i]
            cen_p = (susceptible[i] - censored[i])/susceptible[i]
        else:
            sus_p=1
            cen_p = 1
        survival_function.append(survival_function[i-1]*sus_p)
        censor_function.append(censor_function[i-1]*cen_p)
    
    probability_of_event = [1-i for i in survival_function]
    probability_of_censoring = [1-i for i in censor_function]
    artificial_events = [i>j for i,j in zip(probability_of_event, probability_of_censoring)]
    
    
    for oil_run in grouped_oil:
        oil_run = pd.DataFrame(list(oil_run)[1])
        if artificial_events[oil_run['Idade'].max()-1]:#-1 because 0 index
            df.iat[oil_run['Idade'].idxmax(), df.columns.get_loc('Event')] = 1
    return df
            



def get_oil_rul(df):
    '''
    Determine Remaining Useful Life (RUL) of oil (based on the dados_tratados.csv dataset)
    '''
    df = df.sort_values(['Transformador_ID', 'DataColheita'], ascending = [True,True])
    cur_id = df.iloc[0]['Transformador_ID']
    cur_oil_changes = df.iloc[0]['quantas_trocas_oleo']
    oil_index = df.columns.get_loc('Oil_RUL')
    df['Oil_age'] = 0

    #df_copy = df.copy()
    
    
    for index, row in df.iterrows():
        if index == 0:
            continue
        if row['Transformador_ID'] != df.iloc[index-1]['Transformador_ID']:
            cur_id = row['Transformador_ID']
            #cur_oil_changes = row['quantas_trocas_oleo']
            previous_index = index
        else:
            #if row['quantas_trocas_oleo'] == cur_oil_changes:
            if not row['Event']:
                pass
            else:
                #cur_oil_changes = row['quantas_trocas_oleo'] # should just be +=1
                for i in range(previous_index, index+1):
                    df.iat[i, oil_index] = (df.iloc[index]['DataColheita'] - df.iloc[i]['DataColheita']).days//365
                    #print((df.iloc[index]['DataColheita'] - df.iloc[i]['DataColheita']).days//365)
    
    return df
    df = df[df['Oil_RUL'].apply(lambda x: isinstance(x, int))]
    return df

def feature_importance(rf, feature_list):
    '''
    Uses Random Forest to decide feature importance of dados_tratados.csv dataset
    '''

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


def visualize_feature_importance(rf, feature_list):

    
    importances = rf.feature_importances_
    feature_list.remove('Oil_RUL')
    std = np.std([
    rf.feature_importances_ for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_list)
    
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    plt.show()
    
    return 

def load_impact(df):
    from sklearn.linear_model import Ridge
    
    #df=(df-df.mean())/df.std()
    #model = LinearRegression(normalize=True)
    model = Ridge(alpha=1)
    model.fit(df[['OVER_horas','Q1_horas', 'Q2_horas', 'Q3_horas', 'Q4_horas']], df['Oil_RUL'])
    print([float(i) for i in model.coef_])
    print(float(model.intercept_))

if __name__== "__main__":
    x = load_dados_tratados('asdsad')
    #x = kaplan_meier(x)
    #x = get_oil_rul(x)
    x = plot_censoring_data(x)
    
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(x['Transformador_ID'])
    
    x.drop(['Transformador_ID','DataColheita','quantas_trocas_oleo'],axis=1,inplace=True)
    x.drop(['X2ACF','X5HMF','X5MEF','X2FOL'],axis=1,inplace=True)
    x.dropna(inplace=True)

    x = x.rename({'Idade': 'Age', 'TemperaturaOleo': 'OilTemperature', 'OVER_horas': 'Over_hours', 'Q1_horas':'Q1_hours', 'Q2_horas':'Q2_hours', 'Q3_horas':'Q3_hours', 'Q4_horas':'Q4_hours', 'TIPO TP':'TP Type'}, axis=1)

    
    clf = RandomForestClassifier(max_depth=5, random_state=73)
    
    clf.fit(x.loc[:, x.columns != 'Oil_RUL'],x['Oil_RUL'].astype(int)) # there are some issues with type promotion, presumabely from DataColheita
    
    feature_importance(clf, list(x))
    visualize_feature_importance(clf, list(x))

    #load_impact(x[['OVER_horas','Q1_horas', 'Q2_horas', 'Q3_horas', 'Q4_horas', 'Oil_RUL']])
    # all else equal, impact of load on oil RUL
    
    
