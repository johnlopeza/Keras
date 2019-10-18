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

    sql = open('DataSet/Train2.txt')
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
train.iloc[:, 1] = train.iloc[:, 1].fillna(False)
train.iloc[:, 9] = train.iloc[:, 9].fillna("Sin Definir")


# Definir los parametros de entrenamiento y de prediccion.
x = train.iloc[:, 0:10].values
y = train.iloc[:, [0, 11]].values

# Codificar los datos boleanos ()
labelecoder_x = LabelEncoder()
x[:, 1] = labelecoder_x.fit_transform(x[:, 1])

# agrupar los datos categoricos (localidad y barrio)
onehotencoder = make_column_transformer((OneHotEncoder(sparse=False), [9]), remainder="passthrough")
x = onehotencoder.fit_transform(x)
x = x[:, 1:254]


imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, 252:253])
x[:, 252:253] = imputer.transform(x[:, 252:253]).round(2)

# Separar los Datos de train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# DataFrame
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Se escalan los Datos
sc_X = StandardScaler()
X_train.loc[:, X_train.columns != 244] = sc_X.fit_transform(X_train.loc[:, X_train.columns != 244])
X_test.loc[:, X_test.columns != 244] = sc_X.transform(X_test.loc[:, X_test.columns != 244])

# Inicializar  la RNA (Red Neunora Artificial)
classifier = Sequential()

# Añadir capas de entrada y primera capa Oculta
classifier.add(Dense(units=126, kernel_initializer="uniform", activation="relu", input_dim=252))

# Añadir la segunda capa oculta
classifier.add(Dense(units=126, kernel_initializer="uniform", activation="relu"))

# Añadir la Capa de Salida
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compilar la RNA
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Ajustar la RNA  al conjunto de  Entrenamineto
classifier.fit(x=X_train.loc[:, X_train.columns != 244], y=y_train[:, 1], batch_size=10, epochs=15)

# Predicciones del conjunto de Testing
y_test_pred[2] = classifier.predict(X_test.loc[:, X_test.columns != 244])
y_pred = (y_pred > 0.5)

y_test_pred=np.append(y_test_pred, y_pred, axis=1)
y_test_pred = pd.DataFrame(y_test_pred)


y_test_pred.loc[y_test_pred[0].isin(['5cc37d827cec42002b08a764',
'5ccba9345de703002bc3c2c8',
'5ccc520ac791b50046687294',
'5cccb879424ffe0040d54ae3',
'5ccceae8424ffe0025d59bbc',
'5ccda264f41acf0040b73899',
'5cd15a024a618f001697f9a3',
'5cd185b74c8daa0043c06857',
'5cd1d6b24a618f001398c458',
'5cd2ebae70819e4aff81c14a',
'5cd34546734f770043917d5b',
'5cd42bf4734f7700199284d2',
'5cd489cfc0aad7005e186f45',
'5cd4a987c0aad70016188c72',
'5cd56d14c0aad70034196b1e',
'5cd5db643911930037189854',
'5cd6261f1e0ab9000d3987fd',
'5cd7d6f1d4476000136279e1',
'5cd84fe7b7c305002ba83fb3',
'5cdabd1b2cd44300619a62d6',
'5cf161f63eea2d004cceee3c',
'5cf1741888e702003a727401',
'5cf1b69f68bf5d00584bff7b',
'5cf1c7ac88e702002872a223',
'5cf1dbdb88e702003a72af24',
'5cf3301198dbd1002e022101',
'5cf6db4f7be3130058eabfe7',
'5cf7e3415eef8e000ad97d0b',
'5d1705ad2b7b8d001768f60d',
'5d1fe40f1b2fb6123ec6be53',
'5d29215d923f58002063a3c7',
'5d29fec6923f58001163ff45',
'5d2de9d767af2c2a3be229ee',
'5d30d55cfa03470053ecb4fa',
'5d3c450bc8fe5c002f6f6f57',
'5d4838e5432aa9003367f569',
'5d49ed794890890060366c4f',
'5d4cb8bca8fc2700605c2721'])]





