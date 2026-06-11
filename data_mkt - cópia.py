import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% Analise exploratoria e limpeza de dados


dataset_mkt = pd.read_csv('marketing_campaign_dataset.csv')
dataset_mkt.info()  # resumo do gráfico
dataset_mkt.describe()  # resumo estatistico

# Analisando nulos

nulos = dataset_mkt.isnull().sum()

# Dados duplicados

Duplicados = dataset_mkt.duplicated().sum()

# %% Análise estatistica

# Resumo estatístico: Média, mediana, desvio padrão e types;

numerico_mkt = dataset_mkt.select_dtypes(
    include=(['int64', 'float64']))  # dataframe somente com as variáveis quantitativas

estatistica_mkt = pd.DataFrame({'média': numerico_mkt.mean(),
                                'Mediana': numerico_mkt.median(),
                                'Desvio padrão': numerico_mkt.std(),
                                'Type': numerico_mkt.dtypes})

# %%  Gráfico

# Matriz de correlações de Pearson

matriz_correl = numerico_mkt.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(matriz_correl, annot=True, cmap="seismic", center=0)
# Annot=true (Faz com que os valores apareçam dentro da célula)
# Center=0 (Considere o valor 0 como o ponto central da escala de cores)
# Cmap= escala de cores - 'Blues';'Greens';'Greens';'YlGnBu';'coolwarm';seismic
plt.title('Matriz Correlação de variáveis')
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
plt.figure(figsize=(20, 10))  # Cria área onde os gráficos serão feitos
plt.subplot(1, 2, 1)  # Divide a figura em subgráficos - plt.subplot(linhas, colunas, posição)
sns.histplot(dataset_mkt['CTR'].dropna(), kde=True, bins=25, color='Green')
# .dropna():remove valores nulos, kde=True: suaviza linha ;bins=26: define intervalos do gráficos
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

# Conclusão: Gráfico CTR e CPC apresentam distribuição assimétrica a direita,maioria das campanha possuem CTR baixo;
# em relação ao CPC maioria das campanhas opera com custos de clique relativamente controlados, enquanto algumas campanhas apresentam custos significativamente superiores à média.


# %%  Média de métricas por canal

metricas_canal = dataset_mkt.groupby('Channel_Used').agg({'CTR': ['mean', 'count', 'min', 'max'],
                                                          'CPC': ['mean', 'min', 'max']}).round(2)

print('\
       Métricas por canal:')
print(metricas_canal)

# %% Limpeza de dados pra criação de novos gráficos

dataset_mkt['Acquisition_Cost'] = dataset_mkt['Acquisition_Cost'].replace({
    '\$': '',  # remove o símbolo $
    ',': '',  # remove a vírgula
    ' ': ''  # remove espaços
}, regex=True).astype(float)

# Tranformar ROI em número inteiro

dataset_mkt['ROI'] = pd.to_numeric(dataset_mkt['ROI'], errors='coerce')

# Identificando performance de baixo, médio e alto segmento por quartis
ROI_baixo = dataset_mkt['ROI'].quantile(0.25)
ROI_alto = dataset_mkt['ROI'].quantile(0.75)

Quartis_ROI = pd.DataFrame({'Métrica': ['ROI_baixo', 'ROI_medio', 'ROI_alto'],
                            'ROI': [dataset_mkt['ROI'].quantile(0.25),
                                    dataset_mkt['ROI'].quantile(0.50),
                                    dataset_mkt['ROI'].quantile(0.75)]})
print(Quartis_ROI)

# Categorizando ROI
Boa_performance = dataset_mkt[dataset_mkt['ROI'] >= ROI_alto]
Baixa_performance = dataset_mkt[dataset_mkt['ROI'] <= ROI_baixo]

# %% Gráficos

plt.figure(figsize=(15, 12))

# Análise por canal
plt.subplot(2, 2, 1)
canal_alta_performance = Boa_performance['Channel_Used'].value_counts(normalize=True) * 100
canal_baixa_performance = Baixa_performance['Channel_Used'].value_counts(normalize=True) * 100
pd.DataFrame({'Alta performance': canal_alta_performance,
              'Baixa performance': canal_baixa_performance}).plot(kind='bar', ax=plt.gca())
# kind='bar': qual tipo de gráfico
# ax=plt.gca : o gráfico vai ser feito exatamente no subplot que eu determinei, sem isso o gráfico é feito em qualquer local
plt.title('Análise por canal')
plt.ylabel('Porcentagem')
plt.xticks(rotation=45)  # angulo do onme eixo y

# Distribuição por genero

plt.subplot(2, 2, 3)
Alta_audiencia = Boa_performance['Target_Audience'].value_counts(normalize=True) * 100
Baixa_audiencia = Baixa_performance['Target_Audience'].value_counts(normalize=True) * 100
pd.DataFrame({'Alta performance': Alta_audiencia,
              'Baixa performance': Baixa_audiencia}).plot(kind='bar', ax=plt.gca())
plt.title('Distribuição por gênero')
plt.ylabel('Porcentagem')
plt.xticks(rotation=45)

# Segmantação de usuário
plt.subplot(2, 2, 4)
Alta_segmentacao = Boa_performance['Customer_Segment'].value_counts(normalize=True) * 100
Baixa_segmentacao = Baixa_performance['Customer_Segment'].value_counts(normalize=True) * 100
pd.DataFrame({'Alta performance': Alta_segmentacao,
              'Baixa performance': Baixa_segmentacao}).plot(kind='bar', ax=plt.gca())
plt.title('Segmentação de usuário')
plt.ylabel('Porcentagem')
plt.xticks(rotation=45)

# Campanha
plt.subplot(2, 2, 2)
Campanha_boa = Boa_performance['Campaign_Type'].value_counts(normalize=True) * 100
Campanha_baixa = Baixa_performance['Campaign_Type'].value_counts(normalize=True) * 100
pd.DataFrame({'Alta performance': Campanha_boa,
              'Baixa performance': Campanha_baixa}).plot(kind='bar', ax=plt.gca())
plt.title('Performance por Campanha')
plt.ylabel('Porcentagem')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Resumo gráfico:
print("\
Melhores performances:")
top_performance = Boa_performance.groupby(['Channel_Used', 'Campaign_Type', 'Target_Audience'])['ROI'].agg(
    ['mean', 'count']).sort_values('mean', ascending=False).head(2)
print(top_performance)

print("\
Baixa performance:")
worst_performance = Baixa_performance.groupby(['Channel_Used', 'Campaign_Type', 'Target_Audience'])['ROI'].agg(
    ['mean', 'count']).sort_values('mean', ascending=False).head(2)
print(worst_performance)

# comparação
print("\
Comparação métricas:")
metricas = ['Conversion_Rate', 'CTR', 'CPC', 'ROI']
dados_metrics = pd.DataFrame({
    'Alta performance': Boa_performance[metricas].mean(),
    'Baixa performance': Baixa_performance[metricas].mean()
}).round(2)
print(dados_metrics)


