import cv2
from main import preprocess_img
import numpy as np
import kornia as K
from rot_2 import rotate_image
#from rotate_img import rotate_im
from resizing import resize, get_dims,merge_image
import matplotlib.pyplot as plt
im1='threat_images/BAGGAGE_20170523_094231_80428_B.jpg'
im2='threat_images/BAGGAGE_20170523_085803_80428_D.jpg'


from skimage import io 
from scipy import ndimage   
img=io.imread(im1)
ims=cv2.imread(im1)

rotated = ndimage.rotate(ims, angle=234, mode='nearest')
rotated = cv2.resize(rotated, (ims.shape[:2]))
# rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
# cv2.imwrite('rotated.jpg', rotated)
io.imshow(rotated)
plt.show()
plt.imshow(img)
plt.show()