import pandas as pd
import numpy as np
import sklearn
from sklearn import neural_network

#datele de intrare, separate prin ','
data = pd.read_csv('slump_test.data',sep =',')

#renunt la coloana 'No'
data.drop(['No'], axis=1, inplace=True)

#impart datele si etichetele
date = np.array(data.drop(["SLUMP(cm)","FLOW(cm)","Compressive Strength (28-day)(Mpa)"],1))
etichete = np.array(data[["SLUMP(cm)","FLOW(cm)","Compressive Strength (28-day)(Mpa)"]])

#-------------------------------|||||||||||||||-----------------------------------------------------
print("1 hidden layer, numarul de neuroni pe hidden layer EGAL cu stratul anterior si learning rate 0.1:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes = (7), learning_rate_init= 0.1, max_iter= 10000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

#-------------------------------|||||||||||||||-----------------------------------------------------
print("1 hidden layer, numarul de neuroni pe hidden layer EGAL cu stratul anterior si learning rate 0.01:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes = (7), learning_rate_init= 0.01, max_iter= 10000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

#-------------------------------|||||||||||||||-----------------------------------------------------
print("1 hidden layer, numarul de neuroni pe hidden layer JUMATATE din stratul anterior si learning rate 0.1:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes = (3), learning_rate_init= 0.1, max_iter= 10000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

#-------------------------------|||||||||||||||-----------------------------------------------------
print("1 hidden layer, numarul de neuroni pe hidden layer JUMATATE din stratul anterior si learning rate 0.01:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes = (3), learning_rate_init= 0.01, max_iter= 100000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

#-------------------------------|||||||||||||||-----------------------------------------------------
print("2 hidden layers, numarul de neuroni pe hidden layers EGAL cu stratul anterior si learning rate 0.1:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes=(7,7), learning_rate_init=0.1, max_iter=10000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

#-------------------------------|||||||||||||||-----------------------------------------------------
print("2 hidden layers, numarul de neuroni pe hidden layers EGAL cu stratul anterior si learning rate 0.01:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes=(7,7), learning_rate_init=0.01, max_iter=10000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

#-------------------------------|||||||||||||||-----------------------------------------------------
print("2 hidden layers, numarul de neuroni pe hidden layers JUMATATE din stratul anterior si learning rate 0.1:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes=(3,3), learning_rate_init=0.1, max_iter=10000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

#-------------------------------|||||||||||||||-----------------------------------------------------
print("2 hidden layers, numarul de neuroni pe hidden layers JUMATATE din stratul anterior si learning rate 0.01:\n")
#-------------------------------|||||||||||||||-----------------------------------------------------

#impartire in train si test
date_train, date_test, etichete_train, etichete_test = sklearn.model_selection.train_test_split(date,etichete,test_size = 0.25)

#creez si antrenez MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes=(3,3), learning_rate_init=0.01, max_iter=10000)
regr.fit(date_train,etichete_train)

#fac predictia / Testare
predictii = regr.predict(date_test)

#Eroare patratica medie
suma = 0
for i in range(len(predictii)):
    suma += (etichete_test[i] - predictii[i])**2
print(suma/len(predictii))
print("\n")

