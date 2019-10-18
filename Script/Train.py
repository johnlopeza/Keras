#Paquetes y Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
pd.set_option('mode.chained_assignment', None)

train = pd.read_csv('../DataSet/TrainDataAgo.csv')
train.iloc[:, 20:21] = train.iloc[:, 20:21].fillna(False)
train.iloc[:, 6:7] = train.iloc[:, 6:7].fillna("Sin Definir")
train.iloc[:, 21:22] = train.iloc[:, 21:22].fillna(0)
train["FECHA"] = pd.to_datetime(train["FECHA"])
train["FECHA"]=train["FECHA"].apply(lambda x: x.toordinal())

#Definir los valores de entrenamiento y de prediccion.
x = train.iloc[:, [0,1, 5, 6, 8, 10, 14, 15, 19, 20, 21]].values
y = train.iloc[:, [0,22]].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, [1, 2, 4, 5, 6, 7, 8, 10]])
x[:, [1, 2, 4, 5, 6, 7, 8, 10]] = imputer.transform(x[:, [1, 2, 4, 5, 6, 7, 8, 10]])

labelecoder_x = LabelEncoder()
x[:, 9] = labelecoder_x.fit_transform(x[:,9])
onehotencoder = make_column_transformer((OneHotEncoder(sparse=False), [3]), remainder="passthrough")
x = onehotencoder.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train = pd.DataFrame(X_train)