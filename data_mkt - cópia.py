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

# Conclusão do gráfico: não existe relação entre as variáveis qualitativas

# %% Limpeza de dados e cálculos


# Transformar a variável'Acquisition_Cost' em números, pois seu tipo está em object:
# Primeiro, remover os textos "$",vírgula e espaços em branco

dataset_mkt['Acquisition_Cost'] = dataset_mkt['Acquisition_Cost'].str.replace('[/$,]', '', regex=True).str.strip()
# regex=True - Regex significa Regular Expression (Expressão Regular), é uma forma de procurar padrões em texto
# .strip() - Remove os espaços em branco após o texto

# Segundo, conversão de object para número:

dataset_mkt['Acquisition_Cost'] = pd.to_numeric(dataset_mkt['Acquisition_Cost'], errors='coerce')
# errors='coerce' - transforma valores inválidos em NaN, SUPER IMPORTANTE O COERCE PARA NAO DAR ERRO NOS DADOS


# Cálculo ce CTR e CPC

# CTR = Clicks / Impressões
dataset_mkt['CTR'] = dataset_mkt['Clicks'] / dataset_mkt['Impressions']

# CPC = Custo de aquisição / Clicks

dataset_mkt['CPC'] = dataset_mkt['Acquisition_Cost'] / dataset_mkt['Clicks']

# %% Gráfico de distribuição CTR e CPC

# Gráfico CTR
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.histplot(dataset_mkt['CTR'].dropna(), kde=True, bins=25, color='Green')
plt.title('Distribuição CTR')
plt.xlabel('CTR')
plt.ylabel('Frequência')

# Gráfico CPC
plt.subplot(1, 2, 2)
sns.histplot(dataset_mkt['CPC'].dropna(), kde=True, bins=25, color='Blue')
plt.title('Distribuição CPC')
plt.xlabel('CPC')
plt.ylabel('frequência')

plt.tight_layout()
plt.show()

