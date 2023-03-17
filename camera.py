import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
from email.message import EmailMessage
import ssl
import imghdr
import smtplib


IMG_SIZE = 64

font = cv2.FONT_HERSHEY_SIMPLEX

email_sen='rajpurohitgovind942@gmail.com'
email_password='hbxtiexfjwpsreva'
email_reciever='shuklashivkant14@gmail.com'

subject=' Harm Detected'
body="There has been an accident at"



model1 = AccidentDetectionModel("model.json", 'accident_model_weights.h5')

model2=load_model(r'fire_model.h5')

def startapplication():
    video = cv2.VideoCapture("1.mp4") # for camera use video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))
        
        pred, prob = model1.predict_accident(roi[np.newaxis, :, :])
        print(pred)
        if(pred == "Accident"):
            acc_prob = (round(prob[0][0]*100, 2))
        #else:
        elif(pred == "No Accident"):
            acc_prob=100-(round(prob[0][0]*100, 2))
       # elif(pred == "No Accident"):
        #    prob = (round(prob[1][1]*100, 2))   
            # to beep when alert:
            #if(prob > 99):
        #acc_prob = round(prob[0][0]*100, 2)
        #acc_prob = round((model1.pred(roi)[0][0] * 100),2)



        
        cv2.putText(frame, pred+" "+str(acc_prob), (20, 30), font, 1, (0,255,2), 2)

        roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        image = roi.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        tic = time.time()
        fire_prob = round((model2.predict(image)[0][0] * 100),2)
        toc = time.time()
        #print("Time taken = ", toc - tic)
        #print("FPS: ", 1 / np.float64(toc - tic))
        print("Fire Probability: ", fire_prob)
        print("Predictions: ", model2.predict(image))
        #print(image.shape)
        
        label2 = "Fire Probability: " + str(fire_prob)
        #label1 = pred + str(acc_prob)
        #cv2.rectangle(frame, (0, 0), (280, 40), (255, 255, 255), -1)
        #cv2.putText(frame, label1 + "     " +str(prob) , (10, 25),  font,0.7, (0, 255, 0), 2)
       
        cv2.putText(frame,"                             "+ label2 ,(10,25),font,1,(0,255,2),2)

       ##   em= EmailMessage()
           ## em['To']=email_reciever
          #  em['Subject']=subject
           # em.add_attachment(frame,maintype="image")
          #  em.set_content(body)
          #  context = ssl.create_default_context()
           # with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
           #  smtp.login(email_sen,email_password)
         #    smtp.sendmail(email_sen,email_reciever,em.as_string())
         #   time.sleep(10)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)  


if __name__ == '__main__':
    startapplication()





    





