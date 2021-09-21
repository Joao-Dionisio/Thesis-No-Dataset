# Reading Health index1
#from time import time
import os
import pandas as pd
import csv
#import openpyxl
#from factor_analyzer import FactorAnalyzer
#import seaborn as sns
#import matplotlib.pyplot as plt 



def read_file(file_name, n):
 
    #cur_path = os.path.dirname(__file__)
    #new_path = os.path.relpath('Dados_EDPD\\' + file_name, cur_path)

    '''
    with open("RARI2020.csv", 'r', encoding="utf8") as file:
        csv_file = csv.DictReader(file)
        i = 0
        for row in csv_file:
            print(i)
            i+=1
    '''
    skip = [i for i in range(n)]
    #df = pd.read_excel(new_path, skiprows = skip, usecols = "A,M,S,U,W,Y,AA,AC,AE,AU,AV", skipfooter = 780) # csv reading is faster, but converting to csv seems to add errors
    #df = pd.read_excel(new_path, skiprows = skip, usecols = "A,L,AU, AV, AW") 
    #df.to_csv('RARI2020.csv',encoding='utf-8', index=False)
    df = pd.read_csv('RARI2020.csv')
    '''
    with open('RARI2020.csv') as source:
        rdr = csv.reader(source)
        with open("result","wb") as result:
            wtr = csv.writer(result)
            for r in rdr:
                wtr.writerow((r[0],r[1],r[2],r[3],r[4],r[6],r[7],r[8],r[10],r[11],r[12],r[13],r[14],r[15],r[16],r[18],r[20],r[22],r[24],r[26],r[28],r[32],r[34],r[36],r[38],r[40],r[44],r[45],r[46],r[48],r[50],r[52],r[53],r[54],r[55],r[58],r[59],r[60]))
    '''
    return df



    ##############################

    # Initial check with KMO test to determine if PCA is likely to improve the solution

    #temp = df[['2FAL','CH4','C2H2','C2H4', 'C2H6', 'CO2', 'Tensao_Interfacial','Indice_Acidez','Teor_Agua']]
    
    temp = df[['COR','Massa_Volumica','Tensao_Interfacial','Tensao_Disruptiva','Indice_Acidez','Teor_Agua','Tangente_Delta_90']]
    temp.dropna(inplace=True)
    print(len(temp))
    # Checking if PCA works
    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo_model = calculate_kmo(temp)
    print(kmo_all)
    print(kmo_model)

     
    fa = FactorAnalyzer()
    fa.fit(temp)
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    print(ev)
   
    ############################
    return

    info = []
    with open(new_path, 'r', encoding="utf8") as f: 
        data = []
        for line in f:
            temp = line.split(',')
            if len(temp) <= 27:     # normal reading leads to many errors, working with pandas instead
                data.append(temp)
    
    
    return data



def csv_from_excel():
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath('Dados_EDPD\\Dados_PATH.xlsx', cur_path)
    
    wb = xlrd.open_workbook('Dados_EDPD\\Dados_PATH.xlsx')
    sh = wb.sheet_by_name('Dados_para_PATH')
    your_csv_file = open('Dados_PATH.csv', 'w')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()


if __name__ == "__main__":
    read_file('Dados_PATH.xlsx')
