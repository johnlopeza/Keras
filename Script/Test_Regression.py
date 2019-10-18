#Paquetes y Librerias
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

pd.set_option('mode.chained_assignment', None)

#Carga de Datos
train = pd.read_csv('../DataSet/TrainMejorado2.csv')
train.iloc[:, 20:21] = train.iloc[:, 20:21].fillna(False)
train.iloc[:, 6:7] = train.iloc[:, 6:7].fillna("Sin Definir")
train.iloc[:, 21:22] = train.iloc[:, 21:22].fillna(0)
train["FECHA"] = pd.to_datetime(train["FECHA"])
train["FECHA"]=train["FECHA"].apply(lambda x: x.toordinal())


#Definir los valores de entrenamiento y de prediccion.
x = train.iloc[:, [1, 5, 6, 8, 10, 14, 15, 19, 20, 21]].values
y = train.iloc[:, 22:23].values

#Limpiar los datos
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, [0, 1, 3, 4, 5, 6, 7, 9]])
x[:, [0, 1, 3, 4, 5, 6, 7, 9]] = imputer.transform(x[:, [0, 1, 3, 4, 5, 6, 7, 9]])

#Codificar los datos
labelecoder_x = LabelEncoder()
x[:, 8] = labelecoder_x.fit_transform(x[:,8])
onehotencoder = make_column_transformer((OneHotEncoder(sparse=False), [2]), remainder="passthrough")
x = onehotencoder.fit_transform(x)

#Dividir los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#scalar los Datos
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Regrecion Logistica
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs')
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


