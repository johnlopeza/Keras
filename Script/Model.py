# Paquetes y Librerias
import snowflake.connector as sf
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from Config import Config
from keras.models import Sequential
from keras.layers import Dense
import pickle
from sklearn.metrics import confusion_matrix

pd.set_option('mode.chained_assignment', None)

# conexion a SnowFlake
conn = sf.connect(user=Config.username, password=Config.password, account=Config.account, role=Config.rol,
                  database=Config.database)


# Funsion de ejecucion.
def execute_query(connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    cursor.close()


try:
    sql = 'use {}'.format(Config.database)
    execute_query(conn, sql)

    sql = 'use warehouse {}'.format(Config.warehouse)
    execute_query(conn, sql)

    sql = open('DataSet/Train.txt')
    sql = sql.read()

    cursor = conn.cursor()
    cursor.execute(sql)

    Train = []
    for c in cursor:
        Train.append(c)


except Exception as e:
    print(e)

# DataFrame
train = pd.DataFrame(Train)

# Limpiando Datos

train.iloc[:, 20:21] = train.iloc[:, 20:21].fillna(False)
train.iloc[:, 6:7] = train.iloc[:, 6:7].fillna("Sin Definir")
train.iloc[:, 21:22] = train.iloc[:, 21:22].fillna(0)
train.iloc[:, 1] = pd.to_datetime(train.iloc[:, 1])
train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: x.toordinal())

# Definir los parametros de entrenamiento y de prediccion.
x = train.iloc[:, [0, 1, 5, 6, 8, 10, 14, 15, 19, 20, 21]].values
y = train.iloc[:, [0, 22]].values

# Imputar los datos
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, [1, 2, 4, 5, 6, 7, 8, 10]])
x[:, [1, 2, 4, 5, 6, 7, 8, 10]] = imputer.transform(x[:, [1, 2, 4, 5, 6, 7, 8, 10]]).round(2)

# Codificar los datos
labelecoder_x = LabelEncoder()
x[:, 9] = labelecoder_x.fit_transform(x[:, 9])

onehotencoder = make_column_transformer((OneHotEncoder(sparse=False), [3]), remainder="passthrough")
x = onehotencoder.fit_transform(x)

# Separar los Datos de train y test
X_train, X_test, y_train, y_test = train_test_split(x[:, 1:78], y, test_size=0.2, random_state=0)

# DataFrame
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Se escalan los Datos
sc_X = StandardScaler()
X_train.loc[:, X_train.columns != 67] = sc_X.fit_transform(X_train.loc[:, X_train.columns != 67])
X_test.loc[:, X_test.columns != 67] = sc_X.transform(X_test.loc[:, X_test.columns != 67])

# Inicializar  la RNA (Red Neunora Artificial)
classifier = Sequential()

# Añadir capas de entrada y primera capa Oculta
classifier.add(Dense(units=37, kernel_initializer="uniform", activation="relu", input_dim=76))

# Añadir la segunda capa oculta
classifier.add(Dense(units=37, kernel_initializer="uniform", activation="relu"))

# Añadir la Capa de Salida
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compilar la RNA
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Ajustar la RNA  al conjunto de  Entrenamineto
classifier.fit(x=X_train.loc[:, X_train.columns != 67], y=y_train[:, 1], batch_size=10, epochs=15)

# Predicciones del conjunto de Testing
y_pred = classifier.predict(X_test.loc[:, X_test.columns != 67])



pickle.dump(classifier, open('model.pkl','wb' ))

model = pickle.load( open('model.pkl','rb'))

