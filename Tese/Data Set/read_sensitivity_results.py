# Read sensitivity results and plot them
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("sensitivity_results_20y.txt")
df["param"] = df["param"].str.lstrip('d_primeloadeffectmax_cooling_system_cooling_max_cooling_oil_const=')
df1 = df.iloc[0:11]
df2 = df.iloc[11:21]
df3 = df.iloc[21:32]
df4 = df.iloc[32:42]

all_df = [df1,df2,df3,df4]

for df in all_df:
    plt.plot(df["param"],df["profit"])

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
