import pandas as pd


def main():
    #df = pd.read_excel('Monthly-hourly-load-values_2006-2015.xlsx') 
    #print('as')
    #df.to_csv('Monthly-hourly-load-values_2006-2015.csv',encoding='utf-8', index=False)
    df = pd.read_csv('Monthly-hourly-load-values_2006-2015.csv') 
    df = df[df['Country']=='PT']
    df = df[df['Year']==2015]
    df = df[df['Coverage ratio']==100]
    #df = df.iloc[:,5:]
    df.to_csv('portugal-hourly-load-values_2006-2015.csv',encoding='utf-8', index=False)
    return df.iloc[:,5:]
if __name__ == "__main__":
    df = main()
    print(df.mean())
    #for i in df.mean():
    print(df.mean()/df.mean().min())
    x=df.mean()/df.mean().min()
    print(2*x/x.max())
        #print(i/6030.956164)



'''
[5304.082192,
4918.876712,
4648.019231,
4490.400000,
4413.884932,
4401.013699,
4428.394521,
4583.312329,
5040.109589,
5708.775342,
5969.561644,
6126.534247,
6262.613699,
6087.076712,
6089.545205,
6127.194521,
6068.898630,
6030.956164,
6110.109589,
6278.819178,
6472.964384,
6416.509589,
6225.879452,
5855.778082]
'''
