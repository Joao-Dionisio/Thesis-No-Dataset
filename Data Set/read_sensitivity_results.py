# Read sensitivity results and plot them
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

df = pd.read_csv("sensitivity_results_20y_complete.txt")
df["param"] = df["param"].str.lstrip('Priced_primeloadeffectmax_cooling_system_cooling_max_cooling_oil_const=')
old_df = df

df1 = df.iloc[0:201].astype(float)
df2 = df.iloc[201:402].astype(float)
df3 = df.iloc[402:603].astype(float)
df4 = df.iloc[605:804].astype(float)
df5 = df.iloc[804:].astype(float)
print(df1)
param_names = ["Price","d_prime","load_effect","max_cooling_system_cooling","max_oil_cooling"]
all_df = [df1,df2,df3,df4,df5]


for index, df in enumerate(all_df):
    #df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
    #df.boxplot(column=['profit'])
    if param_names[index] == 'd_prime':
        df = df[df["param"] != 0]
    Q1 = df['profit'].quantile(0.25)
    Q3 = df['profit'].quantile(0.75)
    IQR = Q3- Q1
    #df = df.loc[(df['profit'] >= Q1 - 1.5 * IQR) & (df['profit'] <= Q3 + 1.5 *IQR)]
    #plt.show()
    #reg = LinearRegression().fit(df["param"].values.reshape(-1,1),df["profit"].values.reshape(-1,1))

    #print(reg.score(df["param"].values.reshape(-1,1),df["profit"].values.reshape(-1,1)))
    #y_pred = reg.predict(df["param"].values.reshape(-1,1))
    #plt.plot(df["param"],y_pred)
    plt.xlabel(param_names[index])
    plt.ylabel(df.columns[1])
    plt.scatter(df["param"],df["profit"],s=4,label=param_names[index])
    #plt.xticks(np.linspace(min(df["param"]),max(df["param"]),10))

plt.xticks(np.linspace(0.5,12.5,20))
plt.legend(loc="best")
plt.show()
#plt.show(block=False)
#plt.pause(0.1)
#plt.close()

for i,df in enumerate(all_df):
    plt.boxplot(df['profit'])
    plt.xlabel(param_names[i])
    plt.ylabel("profit")
    plt.show()


'''
with open("sensitivity_results_20y.txt") as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

lines.pop(10)
lines.pop(30)

separate_by_param = []
current_param = ''#lines[0][:10]
same_param = False
for i in range(len(lines)):
    if lines[i][:8] != current_param:
        same_param = False
        current_param = lines[i][:8]
    for j in range(len(lines[i])):
        if lines[i][j] == "=":
            lines[i] = lines[i][j+1:]
            break
    if i%10!=0:
        separate_by_param[-1].append(lines[i])
    else:
        separate_by_param.append([lines[i]])
        same_param = True
            
print(separate_by_param)
for i in separate_by_param:
        i.split(',')
'''
