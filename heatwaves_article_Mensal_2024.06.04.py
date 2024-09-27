#!/usr/bin/env python
# coding: utf-8

# Heatwave Index - New approach

# In[2]:


import pandas as pd
import xarray as xr
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import math
import os


# In[2]:

city="murcia"
#Openfiles
path_t2m_calib = f"/mnt/sda/py_olas/t2m_{city}_1950_2023.nc"
ds_t2m_calib = xr.open_dataset(path_t2m_calib)
ds_t2m_calib["t2m"] = ds_t2m_calib.t2m - 273.15
#Training period
per = ["1960-01-01T00:00:00.000000000", "1990-12-31T23:00:00.000000000"]
ds_t2m_calib_period = ds_t2m_calib.sel(time = slice(per[0], per[1]))
#Daily maximum temperature
ds_t2m_calib_period = ds_t2m_calib_period.resample(time='D').max(dim='time')
#Monthly xarray datasets
month_ds = [ds_t2m_calib_period.sel(time=(ds_t2m_calib_period['time.month'] == i)) for i in range(1,13)] #Armazenando o dataset de cada mês na lista month_ds
#Monthly dataframes
lista_df = []
treinamento_percentil_95 = []
for i in range(0, 12):
    df = month_ds[i].to_dataframe()
    df = df.reset_index()
    df = df[["time", "t2m"]]
    lista_df.append(df)
    percentil_95 = np.percentile(month_ds[i]["t2m"], 95)
    treinamento_percentil_95.append(percentil_95)
    print(percentil_95, i+1)

for df in lista_df:
    df.set_index("time", inplace = True)
    
lista_df[0]


# In[3]:


#Prediction period
#Openfiles
path_t2m_prev = f"/mnt/sda/py_olas/t2m_{city}_1950_2023.nc"
path_r_prev = f"/mnt/sda/py_olas/relum_{city}_1950_2023.nc"

ds_r_prev = xr.open_dataset(path_r_prev)
ds_r_prev["r"] = ds_r_prev.r

ds_t2m_prev = xr.open_dataset(path_t2m_prev)
ds_t2m_prev["t2m"] = ds_t2m_prev.t2m - 273.15
ds_t2m_prev["r"] = ds_r_prev.r

#Prediction period
per_prev = ["1950-01-01T00:00:00.000000000", "2022-12-31T23:00:00.000000000"]
#per_prev = ["1960-01-01T00:00:00.000000000", "1990-12-31T23:00:00.000000000"]
ds_t2m_prev = ds_t2m_prev.sel(time = slice(per_prev[0], per_prev[1]))
month_ds_prev = [ds_t2m_prev.sel(time=(ds_t2m_prev['time.month'] == i)) for i in range(1,13)] #Armazenando o dataset de cada mês na lista month_ds_prev

#CRIAR AS SÉRIES TEMPORAIS PREVISÃO 
lista_df_prev = []

for i in range(0, 12):
    df_prev = month_ds_prev[i].to_dataframe()
    df_prev = df_prev.reset_index()
    df_prev = df_prev[["time", "t2m", "r"]]
    lista_df_prev.append(df_prev)

for df_prev in lista_df_prev:
    df_prev.set_index("time", inplace = True)
    df_prev = df_prev.sort_index()

lista_df_prev[0]


# In[4]:


#Calculation of Heatwave Index
for i in range(0, 12):
    #Extracting data from calibration's t2m
    temps = lista_df[i]['t2m'].values
    #Function for calculate the percentil of a x's temperature
    def calculate_percentil(x):
        return stats.percentileofscore(temps, x)
    
    # Aplicando a função de percentil para cada valor de temperatura no df2
    # Applying the function for each value in lista_df_prev[i]
    lista_df_prev[i]['Target'] = lista_df_prev[i]['t2m'].map(calculate_percentil)
    #Tme/tpe
    lista_df_prev[i]["tpe"] = lista_df_prev[i]["Target"] - 95
    lista_df_prev[i].loc[lista_df_prev[i]["Target"] - 95 <= 0, "tpe"] = 0
    
    #Coef
    lista_df_prev[i]["Coef"] = ((np.exp(lista_df_prev[i]["tpe"])) * (lista_df_prev[i]["r"])) / (1000)
    
    #Heatwave index
    lista_df_prev[i]["HWI"] = ((lista_df_prev[i]["Coef"]) - (0.022)) / (9.7)
    #Se tpe == 0, o índice não deve ser calculado
    lista_df_prev[i].loc[lista_df_prev[i]["tpe"] == 0, "HWI"] = 0
    #Se HWI <= 0.001, não contar como ondas de calor, portanto, HWI é transformado em zero.
    lista_df_prev[i].loc[lista_df_prev[i]["HWI"] <= 0.02, "HWI"] = 0

    #Saving the data in .csv
    result = r"/mnt/sda/py_olas/mensal"
    if not os.path.exists(result):
        os.makedirs(result) 
    lista_df_prev[i].to_csv(f"{result}/{city}_heatwave_ERA5_month_{i+1}.csv")
    lista_df_prev[i].to_excel(f"{result}/{city}_heatwave_ERA5_month_{i+1}.xlsx")


# In[5]:


#testing the script
t2m_95 = lista_df_prev[9].loc[lista_df_prev[9]['Target'] >= 95]
t2m_95.sort_values(by="HWI")


# Ind_prod Calculation

# In[3]:


#Diary Ind_Prod
path1 = r"/mnt/sda/py_olas/mensal"

lista_calib = []

for i in range(1, 13):
    path = f"{path1}/{city}_heatwave_ERA5_month_{i}.csv"
    df = pd.read_csv(path)
    df.reset_index()
    df.set_index("time", inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Grouping the data by day and calculating the sum of the index values
    soma_diaria = df.resample('D')['HWI'].sum()
    # Calculating the total number of hours the index is non-zero for each day
    horas_nao_zero = df['HWI'].apply(lambda x: 1 if x != 0 else 0).resample('D').sum()
    Ind_prod = soma_diaria * horas_nao_zero
    
    t2m = df.resample("D")["t2m"].max()
    r = df.resample("D")["r"].mean()
    
    soma_diaria.name = 'diary_sum'
    horas_nao_zero.name = 'non_zero_hours'
    Ind_prod.name = 'Ind_prod'

    df_I = pd.concat([t2m, r, soma_diaria, horas_nao_zero, Ind_prod], axis=1)

    lista_calib.append(df_I)
    

#lista_calib[0]
Ind_Prod = pd.concat(lista_calib)
Ind_Prod.index = pd.to_datetime(Ind_Prod.index)
Ind_Prod.sort_index(inplace=True)
Ind_Prod.to_csv(f"{result}/{city}_Diary_Ind_Prod_{i}_MAXIMA.csv")
Ind_Prod


# In[12]:


#Monthly Ind_Prod
path1 = r"/mnt/sda/py_olas/mensal"

lista_calib = []

for i in range(1, 13):
    path = f"{path1}/{city}_heatwave_ERA5_month_{i}.csv"
    df = pd.read_csv(path)
    df.reset_index()
    df.set_index("time", inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Grouping the data by day and calculating the sum of the index values
    soma_diaria = df.resample('D')['HWI'].sum()
    # Calculating the total number of hours the index is non-zero for each day
    horas_nao_zero = df['HWI'].apply(lambda x: 1 if x != 0 else 0).resample('D').sum()
    Ind_prod = soma_diaria * horas_nao_zero
    
    t2m = df.resample("D")["t2m"].max()
    r = df.resample("D")["r"].mean()
    
    soma_diaria.name = 'diary_sum'
    horas_nao_zero.name = 'non_zero_hours'
    Ind_prod.name = 'Ind_prod'

    df_I = pd.concat([t2m, r, soma_diaria, horas_nao_zero, Ind_prod], axis=1)

    t2m = df_I.resample("MS")["t2m"].mean()
    r = df_I.resample("MS")["r"].mean()
    Ind_prod_sum = df_I.resample('MS')['Ind_prod'].sum()
    Ind_prod_sum.name = "Ind_prod_monthly_sum"

    hwi = pd.concat([t2m, r, Ind_prod_sum], axis=1)
    hwi = hwi.dropna(how="any")

    lista_calib.append(hwi)
    

Ind_Prod = pd.concat(lista_calib)
Ind_Prod.index = pd.to_datetime(Ind_Prod.index)
Ind_Prod.sort_index(inplace=True)
Ind_Prod.to_csv(f"{result}/{city}_Monthly_Ind_Prod_{i}.csv")
Ind_Prod



# In[ ]:




