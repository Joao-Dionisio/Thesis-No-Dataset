import pandas as pd
#from data_analysis import feature_importance
#from data_analysis import feature_importance2


def read_excel():
    
    #df1 = pd.read_csv("Dados_PATH.csv")
    #df = pd.read_excel("TP ATMT_Avarias (actualização 19Junho2018).xlsx")
    
    #df.to_csv('TP ATMT_Avarias (actualização 19Junho2018).csv',encoding='utf-8', index=False)
    #df = pd.read_excel("RARI2020.xlsx",skiprows=skip)
    #df.to_csv('RARI2020.csv',encoding='utf-8', index=False)

    #df1 = pd.read_csv("RARI2020.csv")
    df1 = pd.read_csv("Dados_PATH.csv")
    df2 = pd.read_csv("Info_manutencao.csv")
    df3 = pd.read_csv("TP ATMT_Avarias (actualização 19Junho2018).csv")
    df3["SAP ID"] = df3["SAP ID"].astype(object)

    df1.sort_values(by="Data Colheita")
    
    #df = df1.merge(df2, how='outer')
    df = df1
    df.rename(columns={'SAPID': 'SAP ID'}, inplace=True)
    df["SAP ID"] = pd.to_numeric(df["SAP ID"], errors="coerce")
    
    df = pd.merge(df, df3[["SAP ID","Ano Avaria"]], on="SAP ID", how="left")
    
    #df = df.drop('Ano fabrico', 1)
    #df = data_cleaning(df)
    #feature_importance(df, "Ano Avaria")
    del df['Transformador']
    del df['N_Serie_LABELEC']
    del df['Data Colheita']
    del df['Diagnostico_furanicos']
    del df['Diagnostico_AC']
    del df['Diagnostico_FQ']
    del df['Recomendacoes']
    del df["Instalacao"]
    
    return df, df3




def predict_fault(df, target):
    from random_forest import random_forest_regression
    from random_forest import random_forest_classification
    #from gaussian_process_classification import gaussian_process_classifier
    from data_analysis import data_cleaning
    
    df = data_cleaning(df)
    
    #df = df.drop_duplicates('SAP ID', keep='last') # Getting just the last records <- might want to sort by date first
    #df.loc[df["Ano Avaria"] == -9999.0, "Ano Avaria"] = 0
    #df.loc[df["Ano Avaria"] != 0, "Ano Avaria"] = 1
    
    df = df[df["Ano Avaria"] != -9999]
    
    #df[target].fillna(0, inplace=True)
    
    #return random_forest_classification(df, target)
    return random_forest_regression(df, target)
    #return gaussian_process_classifier(df, target)
    
if __name__ == "__main__":
    df, df3 = read_excel()
    df = predict_fault(df, "Ano Avaria")
