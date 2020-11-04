# This Python file uses the following encoding: utf-8
# Dataset: total 3168, usei cross validation dividido em 5 partes e cada parte é treinada, male=1584 e female=1584.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.pyplot import title
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

print('Algoritmo de Arvore de Decisao - Com Cross validation')
print('\n')

# Carregando dataset
dataset = pd.read_csv('voice.csv')

#filtro  = dataset['meanfreq'] > 0.070
#dataset  = dataset[filtro]

# Tabela 1
print('Imprimi as 5 primeiras linhas da tabela')
print(dataset.head())

# Apagando algumas colunas para não usar no modelo
#dataset = dataset.drop('Q25',axis=1)
#dataset = dataset.drop('Q75',axis=1)
#dataset = dataset.drop('IQR',axis=1)
#dataset = dataset.drop('skew',axis=1)
#dataset = dataset.drop('kurt',axis=1)
#dataset = dataset.drop('sp.ent',axis=1)
#dataset = dataset.drop('sfm',axis=1)
#dataset = dataset.drop('mode',axis=1)
#dataset = dataset.drop('centroid',axis=1)
#dataset = dataset.drop('meanfun',axis=1)
#dataset = dataset.drop('minfun',axis=1)
#dataset = dataset.drop('maxfun',axis=1)
#dataset = dataset.drop('meandom',axis=1)
#dataset = dataset.drop('mindom',axis=1)
#dataset = dataset.drop('maxdom',axis=1)
#dataset = dataset.drop('dfrange',axis=1)
#dataset = dataset.drop('modindx',axis=1)

features = dataset.columns.difference(['label'])
X = dataset[features].values
y = dataset['label'].values

# Treina o modelo
tree = DecisionTreeClassifier(random_state=1986, criterion='entropy', max_depth=3)
tree.fit(X, y)

#sample1 = [0.059781,  0.064241,  0.032027]
#sample2 = [0.066009,  0.067310,  0.040229]
#sample3 = [0.077316,  0.083829,  0.036718]

sample1 = [0.0597809849598081,0.0642412677031359,0.032026913372582,0.0150714886459209,0.0901934398654331,0.0751219512195122,12.8634618371626,274.402905502067,0.893369416700807,0.491917766397811,0,0.0597809849598081,0.084279106440321,0.0157016683022571,0.275862068965517,0.0078125,0.0078125,0.0078125,0,0]
sample2 = [0.066008740387572,0.0673100287952527,0.040228734810579,0.0194138670478914,0.0926661901358113,0.0732523230879199,22.4232853628204,634.613854542068,0.892193242265734,0.513723842537073,0,0.066008740387572,0.107936553670454,0.0158259149357072,0.25,0.00901442307692308,0.0078125,0.0546875,0.046875,0.0526315789473684]
sample3 = [0.0773155026958227,0.0838294209445061,0.0367184586699814,0.00870105655686762,0.131908017402113,0.123206960845246,30.7571545800584,1024.927704721,0.846389091878782,0.478904979116727,0,0.0773155026958227,0.0987062615673936,0.0156555772994129,0.271186440677966,0.00799005681818182,0.0078125,0.015625,0.0078125,0.0465116279069767]


# Prever as classes de dados
prediction = tree.predict([sample1, sample2, sample3])

print('\n Verifique a precisão do teste com 3 casos: ')
print(prediction)

# Usando Cross Validation
scores_dataset = cross_val_score(tree, X, y, scoring='accuracy', cv=5)

print('\n Acuracia: ')
print(scores_dataset.mean())

# Imprimi todos os gráficos
# Verificar outlines
X1 = dataset.loc[dataset['label'] == 'male']
#print(X1)
Y1 = dataset.loc[dataset['label'] == 'female']
#print(Y1)
fig1 = X1.plot.scatter(x='meanfreq', y='sd', title='Frequencia media x Desvio padrao (male)')
fig2 = Y1.plot.scatter(x='meanfreq', y='sd', title='Frequencia media x Desvio padrao (female)')

plt.show()
#  Acuracia: 0.9608553288344048
#  Acuracia: 0.9588932105765098, tirando o Outlines