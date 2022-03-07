import cv2
import numpy as np
import kornia as K
from torch.utils.data import Dataset

class Process(Dataset):
    def __init__(self,im1,im2):
        self.im1=im1 
        self.im2=im2

    def __getitem__(self):
        img=cv2.imread(self.im1,1)[:,:,::-1]
        target_img=cv2.imread(self.im2,1)[:,:,::-1]
        sift=cv2.SIFT_create()
        standard_kp,standard_des=sift.detectAndCompute(img,None)
        target_kp,target_des=sift.detectAndCompute(target_img,None)
        bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
        matches=bf.match(standard_des,target_des)
        x = list()
        y= list()
        for m in matches:
            (x1, y1) = target_kp[m.trainIdx].pt
            x.append(x1)
            y.append(y1)
        x.sort()
        y.sort()
        x_start = int(x[0])
        x_end = int(x[-1])
        y_start = int(y[0])
        y_end = int(y[-1])
        cropped = target_img[ y_start-20:y_end+15,x_start-10:x_end+40]
    # rot=rotate_image(cropped,40)
        return cropped
im1='threat_images/BAGGAGE_20170523_094231_80428_B.jpg'
im2='threat_images/BAGGAGE_20170523_085803_80428_D.jpg'



import matplotlib.pyplot as plt
c=Process(im1,im2)

plt.imshow(c)
plt.show()