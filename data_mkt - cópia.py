import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

#%% Analise exploratoria e limpeza de dados 


dataset_mkt = pd.read_csv('marketing_campaign_dataset.csv')
dataset_mkt.info()
dataset_mkt.describe()

# Analisando nulos

nulos = dataset_mkt.isnull().sum()

# Dados duplicados

Duplicados = dataset_mkt.duplicated().sum()

#%% Análise estatistica 

# Resumo estatístico: Média, mediana, desvio padrão e types; 

numerico_mkt = dataset_mkt.select_dtypes(include = (['int64','float64']))

estatistica_mkt = pd.DataFrame({ 'média': numerico_mkt.mean(),
                                'Mediana': numerico_mkt.median(),
                                'Desvio padrão': numerico_mkt.std(),
                                'Type': numerico_mkt.dtypes})

# Matriz de correlação 

#correlação_mkt = 

correlation_matrix = numerico_mkt
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.show()