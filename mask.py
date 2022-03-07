import cv2
from main import preprocess_img,clahe
import numpy as np
import kornia as K
from scipy import ndimage 
from rot_2 import rotate_image
#from rotate_img import rotate_im
from resizing import resize, get_dims,merge_image
import matplotlib.pyplot as plt

def segment_and_superimpose(img,bg_img):
    #img=cv2.imread(img,1)[:,:,::-1]
    bg_img=cv2.imread(bg_img,1)[:,:,::-1]
    img=preprocess_img(img,ref_img)
    img[np.where((img == [0, 0, 0]).all(axis = 2))] = [255, 255, 255]


    img=ndimage.rotate(img,45,mode='nearest')
    img=cv2.resize(img,(img.shape[:2]))
    
    #scale down original Image
    img=resize(img)
    image_copy=img
    lower_blue=np.array([180,180,180])
    upper_blue=np.array([255,255,255])
    
    #Normalize Image
    img=img/255
    img=K.image_to_tensor(img)
    #img=K.enhance.equalize(img)
    #img=K.enhance.equalize_clahe(img)   
    img=K.enhance.adjust_gamma(img,2.)
    #img=K.enhance.sharpness(img,200)
    img=K.tensor_to_image(img)
    #renormalize Img
    img=img*255
    mask=cv2.inRange(img,lower_blue,upper_blue)
    masked_image=np.array(image_copy)

    masked_image[mask !=0]=[0,0,0]
    width,height=get_dims(img)
    w,h=get_dims(bg_img)
    w,h=w//2,h//2

    crop_bg=bg_img[h:height+h,w:width+w]
    crop_bg[mask==0]=[0,0,0]
    complete_img=masked_image+crop_bg
    dst=merge_image(bg_img,complete_img,w,h)
    dst1=cv2.fastNlMeansDenoisingColored(dst,None,10,10,7,21)
    return dst1


a='threat_images/BAGGAGE_20170523_094231_80428_B.jpg'
ref_img='threat_images/BAGGAGE_20170522_115645_80428_B.jpg'

b='background_images/S0300542812_20180822020845_L-10_1.jpg'
#img=rotate_image(img,45)
s=segment_and_superimpose(a,b)
plt.imshow(s)
plt.show()


'''
#plt.imshow(complete_img)
#plt.show()
#plt.imshow(masked_image)

#plt.imshow(mask, cmap='gray')
#plt.show()

#cv2.imshow('img', imd)
#plt.imshow(img)
#plt.show()
'''