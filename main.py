import cv2
#from rotate_img import rotate_im
import numpy as np
import kornia as K
from rot_2 import rotate_image
def preprocess_img(im1,im2):
    img=cv2.imread(im1,1)[:,:,::-1]
    target_img=cv2.imread(im2,1)[:,:,::-1]
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

def clahe(img):
    img=img/255
    img=K.image_to_tensor(img)
    #img=K.enhance.equalize(img)
    img=K.enhance.equalize_clahe(img)
    #img=K.enhance.sharpness(img,200)

    img=K.tensor_to_image(img)
    return img
'''
im1='threat_images/BAGGAGE_20170523_094231_80428_B.jpg'
im2='threat_images/BAGGAGE_20170523_085803_80428_D.jpg'

img=preprocess_img(im1,im2)
img[np.where((img == [0, 0, 0]).all(axis = 2))] = [255, 255, 255]
#img[np.where((img==[255,255,255]).all(axis=2))] = [0,0,0]
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




'''