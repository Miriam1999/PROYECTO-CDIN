# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:40:16 2020

@author: moise
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import scipy.spatial.distance as sc
from cdin import CDIN as eda


accidents = pd.read_csv('../data/Accidents_2015.csv')
#%%
my_report = eda.dqr(accidents)

#%%

indx = np.array(accidents.dtypes == 'int64')
col_list = list(accidents.columns.values[indx])

accidents_int = accidents[col_list]
del indx
#%%
my_report_int = eda.dqr(accidents_int)

#%% filtrar el dataset con las columnas  de valores unicos <= 20
indx = np.array(my_report_int.unique_values<=20)
col_list_2 = np.array(col_list)[indx]
accidents_int_unique = accidents_int[col_list_2]
#%% Obtener las medidas de similitud

accidents_dummy = pd.get_dummies(accidents[col_list_2[0]],prefix=col_list_2)

for col in col_list_2[1:]:
    temp=pd.get_dummies(accidents_int_unique[col],prefix=col)
    accidents_dummy = accidents_dummy.join(temp)
del temp
#%% Aplicar indices de similitud
DIST1 =sc.squareform(sc.pdist(accidents_dummy.iloc[0:100,:],'matching'))
temp=pd.DataFrame(DIST1)
#%% Buscar los accidentes más parecidos al accidente que corresponde a el indice 0
D1=temp.iloc[:,0]
D1_sort=np.sort(D1)
D1_index = np.argsort(D1)


















