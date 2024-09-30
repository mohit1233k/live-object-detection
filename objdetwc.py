import cv2
from PIL import Image,ImageTk
import tkinter as tk
import numpy as np


config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model="frozen_inference_graph.pb"

root=tk.Tk()
model=cv2.dnn_DetectionModel(frozen_model,config_file)
model.setInputSize(320,320)
model.setInputScale(1.0/ 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

class_labels=[]
filename="coco.names"
with open (filename ,'rt') as fpt:
    class_labels=fpt.read().splitlines()

# print(class_labels)
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    cap=cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cmarea not opened")
font_scale=1
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,frame=cap.read()
    ClassIndex , confidence, bbox =model.detect(frame ,confThreshold=0.5)
    print(ClassIndex)
    if(len(ClassIndex!=0)):
        for classind,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if(classind!=0):
                cv2.rectangle(frame,boxes,(255,0,0))
                cv2.putText(frame, class_labels[classind - 1], (boxes[0] , boxes[1] -10), font,fontScale=font_scale, color=(0, 0, 255), thickness=2)
       
    cv2.imshow("objdection",frame)
    if cv2.waitKey(25)& 0xff ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()