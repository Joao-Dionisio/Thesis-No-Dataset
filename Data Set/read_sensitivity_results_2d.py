import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pd.options.mode.chained_assignment = None 

df = pd.read_csv("sensitivity_results_2d.txt")

separators = []
separators.extend(df.index[df["Profit"]=="Profit"].tolist())

s_dfs = [df.iloc[0:separators[0]].astype(float)]


for i in range(len(separators)-1):
    cur_df = df.iloc[separators[i]:separators[i+1]]
    cur_df.columns = df.iloc[separators[i]]
    cur_df.drop(separators[i],inplace=True)
    cur_df.reset_index(drop=True,inplace=True)
    s_dfs.append(cur_df.astype(float))

cur_df = df.iloc[separators[-1]:len(df["Profit"])]
cur_df.columns = df.iloc[separators[-1]]
cur_df.drop(separators[-1],inplace=True)
cur_df.reset_index(drop=True,inplace=True)
s_dfs.append(cur_df.astype(float))




for i in s_dfs:
    print('a')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(i.iloc[:,0],i.iloc[:,1],i.iloc[:,2])
    ax.set_xlabel(i.columns[0])
    ax.set_ylabel(i.columns[1])
    ax.set_zlabel(i.columns[2])
    fig.show()
    #input()

#plt.show()
#s_dfs.append(df.iloc[separators[-1]:len(df["Profit"])])
