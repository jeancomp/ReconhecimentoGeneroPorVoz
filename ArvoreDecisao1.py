# This Python file uses the following encoding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#Carregando dataset
dataset = pd.read_csv('voice.csv')

#Apagando algumas colunas para não usar no modelo
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

#Dividi os dados em um conjunto de treinamento e teste
train_features = dataset.iloc[:70,:-1]
test_features = dataset.iloc[70:,:-1]
train_targets = dataset.iloc[:70,-1]
test_targets = dataset.iloc[70:,-1]

#Treina o modelo
tree = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_targets)

#Prever as classes de dados novos e não vistos
prediction = tree.predict(test_features)

#Verifique a precisão
print("A precisao da acuracia: ", "%",tree.score(test_features, test_targets)*100)

#dataset.head(3)
#dataset
#print(dataset)