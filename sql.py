import mysql.connector

cnx = mysql.connector.connect(user='root', database='aqeela_tugas_akhir')
cursor = cnx.cursor()

query = ("SELECT * from diagnosis")

hire_start = datetime.date(1999, 1, 1)
hire_end = datetime.date(1999, 12, 31)

cursor.execute(query)
rows = cursor.fetchall()

for row in rows:
    print row[1]

cursor.close()
cnx.close()