import cv2
cap = cv2.VideoCapture("video.mp4")

obj_det = cv2.createBackgroundSubtractorMOG2(history=100 ,varThreshold=40)


while True:
    ret, frame=cap.read()
    frame=cv2.resize(frame,(540,380),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
    
    height , width, _ =frame.shape
    print(height,width)

    roi=frame[75:380,100:500]
    mask=obj_det.apply(roi)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    

    counters,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in counters: 
        area=cv2.contourArea(cnt)
        if area>100:
            # cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("mask",mask)
    cv2.imshow("Roi",roi)
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(100)
    if key==50:
        break
    
cap.release()
cv2.destroyAllWindows()