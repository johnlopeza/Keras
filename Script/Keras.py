#Paquetes y Librerias
import snowflake.connector as sf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from Config import Config
pd.set_option('mode.chained_assignment', None)

#conexion a SnowFlake
conn = sf.connect(user = Config.username, password = Config.password, account = Config.account, role= Config.rol, database =Config.database)

# Funsion de ejecucion.
def execute_query (connection, query):
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

#Limpiando Datos

train.iloc[:, 20:21] = train.iloc[:, 20:21].fillna(False)
train.iloc[:, 6:7] = train.iloc[:, 6:7].fillna("Sin Definir")
train.iloc[:, 21:22] = train.iloc[:, 21:22].fillna(0)
train.iloc[:, 1] = pd.to_datetime(train.iloc[:, 1])
train.iloc[:, 1]= train.iloc[:, 1].apply(lambda x: x.toordinal())

#Definir los parametros de entrenamiento y de prediccion.
x = train.iloc[:, [0,1, 5, 6, 8, 10, 14, 15, 19, 20, 21]].values
y = train.iloc[:, [0,22]].values

#Imputar los datos
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, [1, 2, 4, 5, 6, 7, 8, 10]])
x[:, [1, 2, 4, 5, 6, 7, 8, 10]] = imputer.transform(x[:, [1, 2, 4, 5, 6, 7, 8, 10]])

#Codificar los datos
labelecoder_x = LabelEncoder()
x[:, 9] = labelecoder_x.fit_transform(x[:,9])
onehotencoder = make_column_transformer((OneHotEncoder(sparse=False), [3]), remainder="passthrough")
x = onehotencoder.fit_transform(x)

#Separar los Datos de train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# DataFrame
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Se escalan los Datos
sc_X = StandardScaler()
X_train.loc[:, X_train.columns != 68] = sc_X.fit_transform(X_train.loc[:, X_train.columns != 68])
X_test.loc[:, X_test.columns != 68]= sc_X.transform(X_test.loc[:, X_test.columns != 68])

# Parametros de TensorFlow
batch_size = 25
x_data = tf.placeholder(shape=[None, 77], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[77,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
y_pred = tf.add(tf.matmul(x_data, A), b)

# Definision del algoritmo
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels= y_target))
pred = tf.nn.sigmoid_cross_entropy_with_logits(logits= y_pred, labels=y_target)
init = tf.global_variables_initializer()
session.run(init)
my_optim = tf.train.GradientDescentOptimizer(learning_rate=0.25)
train_step = my_optim.minimize(loss)


# Clasificador
classification_lr = tf.round(tf.sigmoid(y_pred))
correct_classification = tf.cast(tf.equal(classification_lr, y_target), tf.float32)
accuracy = tf.reduce_mean(correct_classification)
saver = tf.train.Saver()


# Train del Algoritmo
loss_vec = []
train_acc = []
test_acc = []
prediction = []
predictionTest = []

for i in range(1000):
    rand_idx = np.random.choice(len(X_train), size=batch_size)
    rand_x = X_train.loc[rand_idx]
    rand_y = np.transpose([y_train[rand_idx, 1]])
    session.run(train_step, feed_dict={x_data: rand_x.loc[:, rand_x.columns != 68], y_target: rand_y})
    temp_loss = session.run(loss, feed_dict={x_data: rand_x.loc[:, rand_x.columns != 68], y_target: rand_y})
    loss_vec.append(temp_loss)

    temp_acc_train = session.run(accuracy,
                                 feed_dict={x_data: X_train.loc[:, X_train.columns != 68],
                                            y_target: np.transpose([y_train[:, 1]])})
    train_acc.append(temp_acc_train)
    temp_acc_test = session.run(accuracy,
                                feed_dict={x_data: X_test.loc[:, X_test.columns != 68],
                                           y_target: np.transpose([y_test[:, 1]])})
    test_acc.append(temp_acc_test)

    temp_pred = session.run(classification_lr, feed_dict={x_data: X_train.loc[:, X_train.columns != 68]})
    prediction = temp_pred

    temp_predtest = session.run(classification_lr, feed_dict={x_data: X_test.loc[:, X_test.columns != 68]})
    predictionTest = temp_predtest



    if (i + 1) % 100 == 0:
        print("Loss: " + str(temp_loss) + " Acc: " + str(temp_acc_test))

    saver.save(session, 'pred')