import win32com.client
import pythoncom

import runpy
path = "C:\\Users\\Hitesh\\Desktop\\tata_poc\\new_email.txt"
class Handler_Class(object):
    def OnNewMailEx(self, receivedItemsIDs,x=0):
        # RecrivedItemIDs is a collection of mail IDs separated by a ",".
        # You know, sometimes more than 1 mail is received at the same moment.
        for ID in receivedItemsIDs.split(","):
            mail = outlook.Session.GetItemFromID(ID)
            #subject = mail.Subject
            # body = mail.Body
            try:
                # Taking all the "BLAHBLAH" which is enclosed by two "%". 
                # command = re.search(r"%(.*?)%", subject).group(1)
                #print("senders address:" + mail.SenderEmailAddress)
                file = open(path, "w")
                file.write(mail.Subject + " separator " + mail.Body + "this mail has been received from the email: " + mail.SenderEmailAddress)
                file.close
                print("Running the script : tata_poc.py....\n")
                #runpy.run_path('tata_poc.py', run_name="__main__")
                #os.system('python tata_poc.py')
                #print ("Subject : " + subject) # Or whatever code you wish to execute.
                #print ("Body : " + body)
            except:
                pass
            
#runpy.run_path('tata_poc.py', run_name="__main__")
outlook = win32com.client.DispatchWithEvents("Outlook.Application", Handler_Class)

#and then an infinit loop that waits from events.
pythoncom.PumpMessages()
############################################################################################################################################3











