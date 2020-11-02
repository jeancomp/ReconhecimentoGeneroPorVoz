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

# Tabela 1
print('Imprimi a tabela depois da remocao de colunas')
print(dataset.head())

features = dataset.columns.difference(['label'])
X = dataset[features].values
y = dataset['label'].values

# Treina o modelo
tree = DecisionTreeClassifier(random_state=1986, criterion='gini', max_depth=3)
tree.fit(X, y)

sample1 = [0.059781,  0.064241,  0.032027]
sample2 = [0.066009,  0.067310,  0.040229]
sample3 = [0.077316,  0.083829,  0.036718]

# Prever as classes de dados
prediction = tree.predict([sample1, sample2, sample2])

print('\n Verifique a precisão')
print(prediction)

# Usando Cross Validation
scores_dt = cross_val_score(tree, X, y, scoring='accuracy', cv=5)

print('\n Acuracia: ')
print(scores_dt.mean())

# Imprimi todos os gráficos
plt.show()