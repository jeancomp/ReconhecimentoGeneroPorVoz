# This Python file uses the following encoding: utf-8

#Import biblioteca de conjuntos de dados scikit-learn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import plotly.graph_objects as go

print('Algoritmo de Maquina Vetor de Suporte - MVS \n')

# Carregando dataset
dataset = pd.read_csv('voice.csv')

features = dataset.columns.difference(['label'])
X = dataset[features].values
y = dataset['label'].values

# Divida o conjunto de dados em conjunto de treinamento e conjunto de teste
# 70% treino and 30% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)

#Crie um classificador svm
# Linear Kernel
clf = svm.SVC(kernel='linear')

# Treine o modelo usando os conjuntos de treinamento
clf.fit(X_train, y_train)

#Predita a resposta para o conjunto de dados de teste
y_pred = clf.predict(X_test)

# Precisão do modelo: com que frequência o classificador está correto?
print("Acuracia:",metrics.accuracy_score(y_test, y_pred))

#---------------------------------------------------------
count1=0
count2=0

g = sns.PairGrid(dataset[['meanfreq','sd','median','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label")
g = g.map(plt.scatter).add_legend()

#---------------------------------------------------------

plt.show()
# ('Acuracia:', 0.9148264984227129)