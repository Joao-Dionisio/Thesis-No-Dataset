import pandas as pd

# reads maintenance results, selects n best and prints a table 
def tbm_new_sets(n):
    df = pd.read_csv('tbm_results_50y_top5_new.txt')
    #df = pd.read_csv('tbm_results_50y.txt')
    df.sort_values('Profit',ascending=False,inplace=True)
    df = df[df["Profit"] != 0]
    del df["Profit"]
    df = df.reset_index(drop=True)
    print("param: winding_period cooling_system_period ops_period oil_period :=")
    print(df.head(n).to_string(header=False))
    print(';')
    return df.head(n)


def cbm_new_sets(n):
    df = pd.read_csv('cbm_results_50y_top30_degr.txt')
    df.sort_values('Profit',ascending=False,inplace=True)
    df = df[df["Profit"] != 0]
    df = df[df["Winding"] < 1]
    del df["Profit"]
    df = df.reset_index(drop=True)
    print("param: winding_threshold cooling_system_threshold ops_threshold oil_threshold :=")
    print(df.head(n).to_string(header=False))
    print(';')
    return df.head(n)



#df = tbm_new_sets(1)
df = cbm_new_sets(5)
