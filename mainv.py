import cv2
import numpy as np
from time import sleep

width_min=50 #Minimum width of the rectangle
height_min=50 #Minimum height of the rectangle

offset=1 #Allowable error between pixel  

post_line=620 #Count Line Position 

delay=60 #Video FPS

detec = [] #array store the frame detected 
vehicales= 0 #count the detected vehicles

	
def catch_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('videoplayback1080.mp4')
subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) == ord('q'):
        break
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtract.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilate = cv2.morphologyEx (dilat, cv2. MORPH_OPEN , kernel)
    dilate = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    contours,h=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, post_line), (1000, post_line), (200,127,0), 3) 
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_outline = (w >= width_min) and (h >= height_min)
        if not validate_outline:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),3)        
        center = catch_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 9, (0, 0,255), -1)
        
        for (x,y) in detec:
            if y<(post_line+offset) and y>(post_line-offset):
                vehicales+=1
                cv2.line(frame1, (25, post_line), (1000, post_line), (0,127,255), 3)  
                detec.remove((x,y))
                print("car is detected : "+str(vehicales))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(vehicales), (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detect",dilate)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
