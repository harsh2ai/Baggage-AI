import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

from rotate_img import rotate_im
print(os.listdir())
img=cv2.imread('threat_images/BAGGAGE_20170523_094231_80428_B.jpg',1)
target_img=cv2.imread('threat_images/BAGGAGE_20170523_085803_80428_D.jpg',1)

sift=cv2.SIFT_create()
standard_kp,standard_des=sift.detectAndCompute(img,None)
target_kp,target_des=sift.detectAndCompute(target_img,None)
bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
matches=bf.match(standard_des,target_des)


matches = bf.match(standard_des,target_des)
print(matches)
x = list()
y= list()
for m in matches:
  (x1, y1) = target_kp[m.trainIdx].pt
  x.append(x1)
  y.append(y1)
  
x.sort()
y.sort()
print(x)
print(y)
x_start = int(x[0])
x_end = int(x[-1])
y_start = int(y[0])
y_end = int(y[-1])

cropped = target_img[ y_start-20:y_end+15,x_start-10:x_end+40]
print(cropped)

rot=rotate_im(cropped,40)

cv2.imshow('img', rot)
  
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 