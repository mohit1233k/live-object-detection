import cv2
from PIL import Image,ImageTk
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

img1=cv2.imread('image.png')
img=cv2.resize(img1,(640,320))



ClassIndex , confidence, bbox =model.detect(img ,confThreshold=0.5)

print(ClassIndex)
font_scale=3
font=cv2.FONT_HERSHEY_COMPLEX
if len(ClassIndex) > 0:
    for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if conf>0.5:
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
            cv2.putText(img, class_labels[ClassInd - 1], (box[0] , box[1] -10), cv2.FONT_HERSHEY_COMPLEX,fontScale=1, color=(0, 0, 255), thickness=2)
        
cv2.imshow("detection result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
