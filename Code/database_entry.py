

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:47:35 2019

@author: Hitesh
"""

"A prediction system for mails from service now dump files"
"TEXT CLASSIFICATION"






#entity = final.entity_extraction(filename="new_email.txt")
import mysql
"script for creating a ticket"
import mysql.connector
from mysql.connector import errorcode

# my credentials to log in
config = {
  'host':'tata.mysql.database.azure.com',
  'user':'tataAdmin@tata',
  'password':'PIFinance@EY',
  'database':'ad'
}

# Construct connection string
try:
   conn = mysql.connector.connect(**config)
   print("Connection established")
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with the user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cursor = conn.cursor()

  # Drop previous table of same name if one exists
  #cursor.execute("DROP TABLE IF EXISTS inventory;")
  #print("Finished dropping table (if existed).")
  
  
  "to insert the ticket into the required datatable"
###########################################################################################################################################################################
  # Create table
  #cursor.execute("CREATE TABLE inventory (id serial PRIMARY KEY, name VARCHAR(50), quantity INTEGER);")
  #print("Finished creating table.")

  # Insert some data into table
  cursor.execute("INSERT INTO tickets (ticket_number, short_description, long_description, status, assignment_group, affected_ci, date, raised_by_employee_number, assignment_engineer, worklogs, raised_by_name, time, channel) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);", ('10015', 'printer configuration', 'user hitesh called in to open this ticket', 'open', 'ad_service_desk', 'entity_name', '2019-01-05', '8124', 'email_bot', 'please close this ticket as soon as possible', 'email bot', '00:00:00', 'Email'
))
  print("Inserted",cursor.rowcount,"row(s) of data.")
  #cursor.execute("INSERT INTO inventory (name, quantity) VALUES (%s, %s);", ("orange", 154))
  #print("Inserted",cursor.rowcount,"row(s) of data.")
  #cursor.execute("INSERT INTO inventory (name, quantity) VALUES (%s, %s);", ("apple", 100))
  #print("Inserted",cursor.rowcount,"row(s) of data.")

  # Cleanup
  conn.commit()
  cursor.close()
  conn.close()
  print("Done.")










# ticket_number, short_description, long_description, status, assignment_group, affected_ci, date, raised_by_employee_number, assignment_engineer, worklogs, raised_by_name, time, channel
#'9876', 'password reset', 'need to reset system password', 'closed', 'ad_service_desk', 'user system', '2019-01-05', '8124', 'selva', 'please close this ticket as soon as possible', 'voice bot', '00:00:00', 'Email'














