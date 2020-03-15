# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:47:35 2019

@author: Hitesh
"""

"A prediction system for mails from service now dump files"
"TEXT CLASSIFICATION"
from sklearn.externals import joblib
import pickle
#from final import data
import re


my_model = joblib.load('model.pkl')
vec = pickle.load(open("vector.pickel", "rb"))



with open("new_email.txt", "r") as f:
    data = f.read()

#from nltk.stem.wordnet import WordNetLemmatizer
#wnl = WordNetLemmatizer
##need to create an iterable
data = [data]
#data1 = ["""Following mice attacks, caring farmers were marching to Delhi for better living conditions.Delhi police on Tuesday fired water cannons and teargas shells at protesting farmers as they tried to break barricades with their cars, automobiles and tractors."""]

from my_normalisation import normalize_corpus
#from my_normalisation import pos_tag_text
#from nltk.stem import WordNetLemmatizer
#wnl = WordNetLemmatizer
#lemmatized_tokens = [wnl.lemmatize("",word,pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]

####it is taking time to import normalize_corpus
norm_data = normalize_corpus(data, lemmatize=False, tokenize=False)


#coef = my_model.coef_.ravel()

##this is the one with 12,455 features
## using data instead of norm data for now
data_features = vec.transform(data)
#new1 = data_features.toarray()
pred = my_model.predict(data_features)




##entity = final.entity_extraction(filename="new_email.txt")

def get_mail_entity(summary,labels):
    if labels=="Printer":
        printer_num = re.findall(r'[A-Z0-9]{4,10}', summary)
        email = re.findall(r'\S+@\S+', summary)
        return printer_num,email


#os.system("python testmail.py")


#df = pd.read_excel('Oct-18_Call Dump.xlsx')
#df_printer = df[df['Sub-Category']=='Printer']


#######################################################################################################################################################3333
"DATABASE ENTRY COMPLETED"

import mysql
"script for creating a ticket"
import mysql.connector
from mysql.connector import errorcode


"building a function to connect to the database"
def ticket(entity):
    config = {
  'host':'tata.mysql.database.azure.com',
  'user':'tataAdmin@tata',
  'password':'PIFinance@EY',
  'database':'ad'
  }
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
    
    # Insert some data into table
    cursor.execute("INSERT INTO tickets (ticket_number, short_description, long_description, status, assignment_group, affected_ci, date, raised_by_employee_number, assignment_engineer, worklogs, raised_by_name, time, channel) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);", ('10050', 'printer configuration', 'user hitesh called in to open this ticket', 'Assigned', 'EY_Emailbot', entity, '2019-01-05', '8124', 'Hitesh', 'please close this ticket as soon as possible', 'email bot', '00:00:00', 'Email'))

    print("Inserted",cursor.rowcount,"row(s) of data.")

    conn.commit()
    cursor.close()
    conn.close()
    print("Done.")
    return True
    
import smtplib
def send_notification(email_id):
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login("eymailpoc@gmail.com", "eighteight88") 

    # message to be sent 
    message = "Your ticket has been raised successfully"

    # sending the mail
    # to the receivers id
    s.sendmail("eymailpoc@gmail.com", email_id[0], message) 
    print("mail sent")
    # terminating the session 
    s.quit()

############################################################################################################################################
"need to create automatic ticket number"
"mail must be sent to those who raised the ticket"





if __name__ == '__main__':
    print_num,email_id = get_mail_entity(data[0],labels=pred)
    ticket_generated = ticket(print_num[0])
    if ticket_generated :
        send_notification(email_id)
        
    

































