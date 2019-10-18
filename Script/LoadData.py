import snowflake.connector as sf
from Config import Config

conn = sf.connect(user = Config.username, password = Config.password, account = Config.account, role= Config.rol, database =Config.database)
'''Carga de datos Train'''
def execute_query (connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    cursor.close()

try:

    sql = 'use {}'.format(Config.database)
    execute_query(conn,sql)

    sql = 'use warehouse {}'.format(Config.warehouse)
    execute_query(conn,sql)

    sql = open('/Users/pro2017/Documents/Predictions/TensorFlow/DataSet/Train.txt', 'r')
    sql = sql.read()

    cursor = conn.cursor()
    cursor.execute(sql)

    Train = []
    for c in cursor:
        Train.append(c)

except Exception as e:
    print(e)






