# This Python file uses the following encoding: utf-8
# Dataset: total 3168, 70% = 2218 e 30% = 950, male=1584 e female=1584.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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
sns.heatmap(dataset.corr(), annot=True, cmap="Blues")

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
print("Acuracia: ", "%",tree.score(test_features, test_targets)*100)

print('\n Teste com 3 casos-Masculino')
#sample1 = [0.059781,  0.064241,  0.032027]
#sample2 = [0.066009,  0.067310,  0.040229]
#sample3 = [0.077316,  0.083829,  0.036718]

sample1 = [0.0597809849598081,0.0642412677031359,0.032026913372582,0.0150714886459209,0.0901934398654331,0.0751219512195122,12.8634618371626,274.402905502067,0.893369416700807,0.491917766397811,0,0.0597809849598081,0.084279106440321,0.0157016683022571,0.275862068965517,0.0078125,0.0078125,0.0078125,0,0]
sample2 = [0.066008740387572,0.0673100287952527,0.040228734810579,0.0194138670478914,0.0926661901358113,0.0732523230879199,22.4232853628204,634.613854542068,0.892193242265734,0.513723842537073,0,0.066008740387572,0.107936553670454,0.0158259149357072,0.25,0.00901442307692308,0.0078125,0.0546875,0.046875,0.0526315789473684]
sample3 = [0.0773155026958227,0.0838294209445061,0.0367184586699814,0.00870105655686762,0.131908017402113,0.123206960845246,30.7571545800584,1024.927704721,0.846389091878782,0.478904979116727,0,0.0773155026958227,0.0987062615673936,0.0156555772994129,0.271186440677966,0.00799005681818182,0.0078125,0.015625,0.0078125,0.0465116279069767]

prediction = tree.predict([sample1, sample2, sample2])
print(prediction)

# Imprimi todos os gráficos
plt.show()
# Acuracia:  96.88908926961226