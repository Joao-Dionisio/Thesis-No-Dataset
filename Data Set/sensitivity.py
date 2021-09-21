import pandas as pd

df = pd.read_csv('tbm_results_20y_top1_chi.txt', delim_whitespace=True)
df2 = pd.read_csv('tbm_results_20y_top1_q.txt', delim_whitespace=True)
df2.drop(['1','3','5','7','9'],axis=1,inplace=True)
df2 = df2['2'].append(df2['4']).append(df2['6']).append(df2['8']).append(df2['10'])

print(len(df['OPS'])-1)
for col in df:
    txt = ''
    for i in df[col]:
        txt += str(i) + ' '
    print(txt)

print()
print()
txt=''
for i in df2:
    txt += str(i) + ' '
print(txt)
    
