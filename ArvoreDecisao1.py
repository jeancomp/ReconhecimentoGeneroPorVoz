# This Python file uses the following encoding: utf-8
# Dataset: total 3168, 70% = 2218 e 30% = 950, male=1584 e female=1584.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

print('Algoritmo de Arvore de Decisao')
print('\n')

# Carregando dataset
dataset = pd.read_csv('voice.csv')

# Figura 1
# Total genero masculino x feminino
#print(dataset['label'].value_counts())
plt.title('Total genero masculino x feminino')
dataset['label'].value_counts().plot(kind='pie')

# Tabela 1
print('Imprimi as 5 primeiras linhas da tabela')
print(dataset.head())

# Apagando algumas colunas para não usar no modelo
dataset = dataset.drop('Q25',axis=1)
dataset = dataset.drop('Q75',axis=1)
dataset = dataset.drop('IQR',axis=1)
dataset = dataset.drop('skew',axis=1)
dataset = dataset.drop('kurt',axis=1)
dataset = dataset.drop('sp.ent',axis=1)
dataset = dataset.drop('sfm',axis=1)
dataset = dataset.drop('mode',axis=1)
dataset = dataset.drop('centroid',axis=1)
dataset = dataset.drop('meanfun',axis=1)
dataset = dataset.drop('minfun',axis=1)
dataset = dataset.drop('maxfun',axis=1)
dataset = dataset.drop('meandom',axis=1)
dataset = dataset.drop('mindom',axis=1)
dataset = dataset.drop('maxdom',axis=1)
dataset = dataset.drop('dfrange',axis=1)
dataset = dataset.drop('modindx',axis=1)

# Tabela 2
print('Imprimi a tabela, depois da remoção de algumas linhas')
print(dataset.head())

# Dividi os dados em um conjunto de treinamento e teste
train_features = dataset.iloc[:2218,:-1]
test_features = dataset.iloc[950:,:-1]
train_targets = dataset.iloc[:2218,-1]
test_targets = dataset.iloc[950:,-1]

# Figura 2
# Correlação das variaveis com um mapa de calor
# Pra tirar algumas conlusões
# 1 -
# 2 -
#plt.title('Correlação das variaveis com um mapa de calor')
plt.figure(figsize=(10,10))
sns.heatmap(train_features.corr(), annot=True, cmap="Blues")

# Figura 3
# Plotando os dados no grafico barra
plt.figure(figsize=(9,6))
sns.barplot(x='meanfreq',y='sd',data=train_features)

# Treina o modelo
tree = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_targets)

# Prever as classes de dados novos e não vistos
prediction = tree.predict(test_features)

# Imprimir os valores da Predicao
#dataset['resulPredicao'] = (prediction)
#print(dataset)

# Verifique a precisão
print("A precisao da acuracia: ", "%",tree.score(test_features, test_targets)*100)

print('\n Teste com 3 casos-Masculino')
sample1 = [0.059781,  0.064241,  0.032027]
sample2 = [0.066009,  0.067310,  0.040229]
sample3 = [0.077316,  0.083829,  0.036718]
prediction = tree.predict([sample1, sample2, sample2])
print(prediction)

# Imprimi todos os gráficos
plt.show()