# trying to plot evolution of solution

from matplotlib import pyplot as plt
import pandas as pd

pdm=pd.read_csv('final_model_results_50y_new.txt',sep='\s+')
#pdm=pd.read_csv('final_model_results_100y_gap.txt',sep='\s+')
cbm=pd.read_csv('cbm_results_50y_top1_new.txt',sep='\s+')
tbm=pd.read_csv('tbm_results_50y_top1_new.txt',sep='\s+')
#cbm=pd.read_csv('cbm_results_20y_top1.txt',sep='\s+')
#tbm=pd.read_csv('tbm_results_20y_top1.txt',sep='\s+')



pdm['Time'] = pd.to_numeric(pdm['Time'].str.rstrip('s'))
pdm = pdm.replace({'H':''}, regex=True)
cbm['Time'] = pd.to_numeric(cbm['Time'].str.rstrip('s'))
cbm = cbm.replace({'H':''}, regex=True)
tbm['Time'] = pd.to_numeric(tbm['Time'].str.rstrip('s'))
tbm = tbm.replace({'H':''}, regex=True)


pdm['index'] = pdm.index
pdm['BestBd'] = pd.to_numeric(pdm['BestBd'], errors='coerce')
pdm['Incumbent'] = pd.to_numeric(pdm['Incumbent'], errors='coerce')
cbm['index'] = cbm.index
cbm['BestBd'] = pd.to_numeric(cbm['BestBd'], errors='coerce')
cbm['Incumbent'] = pd.to_numeric(cbm['Incumbent'], errors='coerce')
tbm['index'] = tbm.index
tbm['BestBd'] = pd.to_numeric(tbm['BestBd'], errors='coerce')
tbm['Incumbent'] = pd.to_numeric(tbm['Incumbent'], errors='coerce')



#pdm['Index'] = range(0, len(df))
#plt.gca().invert_yaxis()
pdm_bestbd = pdm['BestBd'].tolist()
cbm_bestbd = cbm['BestBd'].tolist()
tbm_bestbd = tbm['BestBd'].tolist()
#plt.scatter(pdm['Time'][:10],pdm['Incumbent'][:10],s=1)
#plt.scatter(pdm['Time'][:10],pdm['BestBd'][:10],s=1)

#print(len(pdm["Time"]))
pdm_plot = plt.subplot()
#for i in range(40, len(pdm["Time"])-1):
#    plt.plot([pdm["Time"].iloc[i],pdm["Time"].iloc[i+1]],[pdm["Incumbent"].iloc[i],pdm["Incumbent"].iloc[i+1]],'ro-',lw=0.3, markersize=1)
pdm_plot.scatter(pdm['Time'][130:],pdm['Incumbent'][130:],s=1,label='PdM Incumbent') # initial solutions mess up the graph
pdm_plot.plot(pdm['Time'][130:],pdm['Incumbent'][130:]) # initial solutions mess up the graph
#pdm_plot.scatter(pdm['Time'],pdm['BestBd'],s=1,label='PdM BestBd')

#plt.show()

cbm_plot = plt.subplot()
cbm_plot.scatter(cbm['Time'][40:],cbm['Incumbent'][40:],s=1,label='CbM Incumbent') # initial solutions mess up the graph
cbm_plot.plot(cbm['Time'][40:],cbm['Incumbent'][40:]) # initial solutions mess up the graph
#cbm_plot.scatter(cbm['Time'],cbm['BestBd'],s=1,label='CbM BestBd')
cbm_plot.legend(loc='lower right')

tbm_plot = plt.subplot()
tbm_plot.scatter(tbm['Time'][50:],tbm['Incumbent'][50:],s=1,label='TbM Incumbent') # initial solutions mess up the graph
tbm_plot.plot(tbm['Time'][50:],tbm['Incumbent'][50:]) # initial solutions mess up the graph
#tbm_plot.scatter(tbm['Time'],tbm['BestBd'],s=1,label='CbM BestBd')
tbm_plot.legend(loc='lower right')

#plt.scatter(df['Time'],bestbd,s=1)

plt.xlabel('Time (s)')
plt.ylabel('Profit (thousands of €)')
pdm_plot.legend(loc='best')
#plt.savefig("Media/pdm_cbm_tbm_50y_new.pdf", bbox_inches="tight")
plt.show()




# Manually changed column names and stuff. We only care about Incumbent, BestBd and Time

#    Nodes    |    Current Node    |     Objective Bounds      |     Work
# Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

# Nodes Current_Node  Objective_Bounds Work  Incumbent BestBd   Gap  Depth Time         
