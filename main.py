# packages import
import numpy as np
import pandas as pd
import cmath
import random
from scipy.stats import gamma
from os import system, name 
"""
Function definitions
"""
def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

def calculate_df(dev_fact, dataframe):
  j = 2 # column counter
  while j <= len(dataframe):
    # initialization
    sum1 = 0
    sum2 = 0
    for i in range(0, len(dataframe)): # row counter
      if np.isfinite(dataframe.at[i, f'{j}']):
        sum1 = sum1 + dataframe.at[i, f'{j-1}']
        sum2 = sum2 + dataframe.at[i, f'{j}']
    dev_fact.append(sum2/sum1) # we take only the 6 digits after the ,
    j = j+1

def fill_na(column1, column2, dev_factor):
  for i in range(0, len(column2)):
    if np.isnan(column2[i]):
      column2[i] = column1[i] * dev_factor

def cum_data(data):
  for i, item in data.iterrows():
      data.at[i]  = pd.Series(item).cumsum()

def lower_triangle(data,dev_fact):
  i = 2
  while i <= len(data):
    fill_na(data[f'{i-1}'], data[f'{i}'], dev_fact[i-2])
    i = i + 1

def incremental(data):
  for i, item in data.iterrows():
    for j in range(len(item)-1, 0, -1):
      item[j] = item[j] - item[j-1]

def deviance(returned_data, x_data, y_data):
  for i in range(0,len(x_data)):
    for j in range(0,len(x_data)):
      if i == 1 and j == 10:
        returned_data.iat[i,j] = 0
      else:
        expression = (x_data.iat[i,j]*cmath.log(x_data.iat[i,j],2))-(x_data.iat[i,j]*cmath.log(y_data.iat[i,j],2))-x_data.iat[i,j]+y_data.iat[i,j]
        returned_data.iat[i,j] = 0 if np.isnan(expression) else expression

def chi_squared(x):
  return x*x

def sum_diagonal(data):
  data_sum = 0
  nb = len(data)+1
  for i in range(0,len(data)):
    nb=nb-1
    for j in range(0,nb):
      data_sum = data_sum + data.iat[i,j]
  return data_sum

def gamma_error(data, peardis):
  for i in range(len(data)-1, -1, -1):
    for j in range(2, len(data)):
      if int(data.iat[i,j]) != 0:
        a = (np.absolute(data.iat[i,j])/peardis) # shape
        elem = gamma.rvs(a,scale=peardis, size=1)
        data.iat[i,j] = elem[0]
      else:
        data.iat[i,j] = 0

def reserve_calcul(reserve_vector, data):
  somme = 0
  K = 10
  for i in range(1, len(data)):
    reserve = 0
    for j in range(len(data)-1, K+1, -1):
      reserve = reserve+data.iat[i,j]
    reserve_vector.append(reserve)
    K = K-1
    somme = somme + reserve
  reserve_vector.append(somme)
# define the initial data
"""
This is where we define the initial dataframe to use on our prediction process with the method "Chain ladder"
"""
initial_data = pd.read_csv("data.csv")
print("-"*15, "Initial Data", "-"*15)
print(initial_data)
# build the cumulative data
print("-"*15, "Cumulative Data", "-"*15)
cum_data(initial_data)
print(initial_data)
# get the development factors
dev_factors = []
calculate_df(dev_factors, initial_data)
print("-"*15, "Development Factors", "-"*15)
print(dev_factors)
# build the lower triangle of the matrix
lower_triangle(initial_data, dev_factors)
print("-"*15, "Fitted Cumulative", "-"*15)
print(initial_data)
#provision
dev_provision = []
i = 0
while i < len(initial_data):
  print(initial_data.iat[i,len(initial_data)-i-1])
  dev_provision.append(initial_data.iat[i,len(initial_data)-1] - initial_data.iat[i,len(initial_data)-i-1])
  i = i + 1
print("-"*15, "Provision" ,"-"*15)
print(dev_provision)
print("-"*15, "Provision Total" ,"-"*15)
print(sum(dev_provision))
# Fitted incremental
fitted_inc_data = initial_data.copy()
incremental(fitted_inc_data)
print("-"*15, "Fitted Incremental", "-"*15)
print(fitted_inc_data)
'''
i.i.d
X: Valeur avant "Fitted Incremental"
X~: Valeur apres "Fitted Incremental"
r(i,j)=(X(i,j)-X~(i,j))/sqrt(X~(i,j))
i+j<I
'''
# Pearson residuals
pearson_resid_data = initial_data.copy()
I = len(pearson_resid_data)
print("Length", I)
for i, item in pearson_resid_data.iterrows():
  # I = I - 1
  for j in range(I-1, -1, -1):
    # if i+j < I:
    item[j]=(item[j]-fitted_inc_data.iat[i, j])/cmath.sqrt(fitted_inc_data.iat[i, j])
print("-"*15, "Pearson Residuals", "-"*15)
print(pearson_resid_data)

'''
Création d'un vecteur pour avoir les valeurs plus facilement
Il est possible d'ajuster les résidus de Pearson en corrigeant chaque terme par le nombre de degrés de liberté.
Pour ce faire, chaque résidu est multiplié par le facteur
DoF=sqrt(B/B-p) avec
B=(I*(I+1)/2)-2 : nombre des résidus non exclus
et 
p: nombre de paramètres
Si I=10, B = 53
et p = 19
'''
B=((I*(I+1))/2)-2
p=(2*I)-1
dof_factor = cmath.sqrt(B/(B-p))
print("B=",B ,"p=", p, "DoF=", dof_factor)
resampled_initial_vector = []
for i, item in pearson_resid_data.iterrows():
  for j in range(0,I,1):
    resampled_initial_vector.append(item[j]*dof_factor)

# Ré-échantillonnage
def bootstrap_iteration(debug):
  '''
  Création d'une nouvelle matrice aléatoirement
  '''
  resampled_matrix = initial_data.copy()
  for i, item in resampled_matrix.iterrows():
    for j in range(0,I,1):
      item[j]=resampled_initial_vector[random.randint(0, len(resampled_initial_vector)-1)]
  if debug == True:
    print("-"*15,"Resampled Matrix","-"*15)
    print(resampled_matrix)
    print("-"*15,"Resampled Cumulative Matrix","-"*15)
  resampled_cumulative_matrix = resampled_matrix.copy()
  cum_data(resampled_cumulative_matrix)
  if debug == True:
    print(resampled_cumulative_matrix)
    print("-"*15,"Resampled Cumulative Matrix Dev Factors","-"*15)
  resampled_df = []
  calculate_df(resampled_df, resampled_cumulative_matrix)
  if debug == True:
    print(resampled_df)
    print("-"*15,"Forecast Resampled Cumulative Matrix","-"*15)
  lower_triangle(resampled_cumulative_matrix, resampled_df)
  if debug == True:
    print(resampled_cumulative_matrix)
    print("-"*15,"Forecast Resampled Incremental Matrix","-"*15)
  resampled_incremental_matrix = resampled_cumulative_matrix.copy()
  incremental(resampled_incremental_matrix)
  if debug == True:
    print(resampled_incremental_matrix)
  '''
  Deviance Ref: https://en.wikipedia.org/wiki/Deviance_(statistics)
  TLDR; deviance is a goodness-of-fit statistic for a statistical model
  On va travailler avec le deviance unitaire de la distribution de la loi de Poisson (Ref: Examples)
  '''
  deviance_matrix = resampled_incremental_matrix.copy()
  deviance(deviance_matrix, resampled_incremental_matrix, resampled_cumulative_matrix)
  if debug == True:
    print("-"*15,"Statistique Deviance","-"*15)
    print(deviance_matrix)
  '''
  Quote: "an approximate chi-squared distribution with k-degrees of freedom. This can be used for hypothesis testing on the deviance."
  '''
  chi_sq = deviance_matrix.copy()
  chi_sq.applymap(chi_squared)
  if debug == True:
    print("-"*15,"Statistique Deviance Test - Chi Squared","-"*15)
    print(chi_sq)
  chi_sq_sum = sum_diagonal(chi_sq)
  pearson_dist = chi_sq_sum/(B-p)
  if debug == True:
    print("-"*15,"Statistique Deviance Test - Chi Squared Sum","-"*15)
    print(chi_sq_sum)
    print("-"*15,"Pearson Chi Squared","-"*15)
    print(pearson_dist)
  '''
  Application de la loi gamma pour mesurer l'erreur
  '''
  loi = 2
  gamma_matrix = resampled_incremental_matrix.copy()
  gamma_error(gamma_matrix, pearson_dist)
  if debug == True:
    print("-"*15,"Gamma Error","-"*15)
    print(gamma_matrix)
  '''
  Calc reserve
  '''
  Reserve = []
  reserve_calcul(Reserve, gamma_matrix)
  if debug == True:
    print("-"*15,"Reserve Vector","-"*15)
    print(Reserve)
  return Reserve

def bootstrap(initial_data, simulations, debug):
  index = range(0, simulations)
  columns = range(1, len(initial_data)+1)
  bootstrap_matrix = pd.DataFrame(index=index, columns=columns)
  bootstrap_matrix.fillna(0)
  for i in range(0, simulations):
    percent = (i/simulations)
    clear()
    print("Simulation Progress:")
    print(f'{percent} % Done')
    reserve = bootstrap_iteration(debug)
    for j in range(0, 10):
      bootstrap_matrix.iat[i, j] = reserve[j]
  return bootstrap_matrix

BM = bootstrap(initial_data, 1000, False)
print("-"*15,"Bootstrap Simulation","-"*15)
print(BM)
print("Mean")
mean_vector = BM.mean()
print(len(mean_vector))
print(mean_vector.sum()/len(mean_vector))
print("Standard Deviation")
std_vector = BM.std()
print(std_vector.sum()/len(std_vector))
print("-"*15,"End Bootstrap Simulation","-"*15)