# This Python file uses the following encoding: utf-8
# Dataset: total 3168, usei cross validation dividido em 5 partes e cada parte é treinada, male=1584 e female=1584.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.pyplot import title
from sklearn import svm
from sklearn.model_selection import cross_val_score

print('Algoritmo de Maquina Vetor de Suporte - MVS')
print('\n')

# Carregando dataset
dataset = pd.read_csv('voice.csv')

#filtro  = dataset['meanfreq'] > 0.070
#dataset  = dataset[filtro]

# Tabela 1
print('Imprimi as 5 primeiras linhas da tabela')
print(dataset.head())

features = dataset.columns.difference(['label'])
X = dataset[features].values
y = dataset['label'].values

# Treina o modelo - MVS
clf = svm.SVC()
clf.fit(X, y)
#SVC()

sample1 = [0.0597809849598081,0.0642412677031359,0.032026913372582,0.0150714886459209,0.0901934398654331,0.0751219512195122,12.8634618371626,274.402905502067,0.893369416700807,0.491917766397811,0,0.0597809849598081,0.084279106440321,0.0157016683022571,0.275862068965517,0.0078125,0.0078125,0.0078125,0,0]
sample2 = [0.066008740387572,0.0673100287952527,0.040228734810579,0.0194138670478914,0.0926661901358113,0.0732523230879199,22.4232853628204,634.613854542068,0.892193242265734,0.513723842537073,0,0.066008740387572,0.107936553670454,0.0158259149357072,0.25,0.00901442307692308,0.0078125,0.0546875,0.046875,0.0526315789473684]
sample3 = [0.0773155026958227,0.0838294209445061,0.0367184586699814,0.00870105655686762,0.131908017402113,0.123206960845246,30.7571545800584,1024.927704721,0.846389091878782,0.478904979116727,0,0.0773155026958227,0.0987062615673936,0.0156555772994129,0.271186440677966,0.00799005681818182,0.0078125,0.015625,0.0078125,0.0465116279069767]


# Prever as classes de dados
prediction = clf.predict([sample1,sample2,sample3])

print('\n Verifique a precisão do teste com 3 casos: ')
print(prediction)

# Usando Cross Validation
scores_dataset = cross_val_score(clf, X, y, scoring='accuracy', cv=5)

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
#  Acuracia: 0.6748452661422354