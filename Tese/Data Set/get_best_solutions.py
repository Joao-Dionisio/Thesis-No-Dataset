import pandas as pd

# reads maintenance results, selects n best and prints a table 
def tbm_new_sets(n):
    #df = pd.read_csv('tbm_results_20y.txt')
    #df = pd.read_csv('tbm_results_50y_top5.txt')
    df.sort_values('Profit',ascending=False,inplace=True)
    df = df[df["Profit"] != 0]
    del df["Profit"]
    df = df.reset_index(drop=True)
    print("param: winding_period cooling_system_period ops_period oil_period :=")
    print(df.head(n).to_string(header=False))
    print(';')
    return df.head(n)


def cbm_new_sets(n):
    df = pd.read_csv('cbm_results_50y_top30(1).txt')
    df.sort_values('Profit',ascending=False,inplace=True)
    df = df[df["Profit"] != 0]
    del df["Profit"]
    df = df.reset_index(drop=True)
    print("param: winding_threshold cooling_system_threshold ops_threshold oil_threshold :=")
    print(df.head(n).to_string(header=False))
    print(';')
    return df.head(n)



#df = tbm_new_sets(30)
df = cbm_new_sets(5)
