import cv2
import numpy as np

def rotate_im(image, angle):
    image_height = image.shape[0]
    image_width = image.shape[1]
    diagonal_square = (image_width*image_width) + (
        image_height* image_height
    )
    #
    diagonal = round(np.sqrt(diagonal_square))
    padding_top = np.round((diagonal-image_height) // 2)
    padding_bottom = np.round((diagonal-image_height) // 2)
    padding_right = np.round((diagonal-image_width) // 2)
    padding_left = np.round((diagonal-image_width) // 2)
    padded_image = cv2.copyMakeBorder(image,
                                      top=padding_top,
                                      bottom=padding_bottom,
                                      left=padding_left,
                                      right=padding_right,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0
            )
    padded_height = padded_image.shape[0]
    padded_width = padded_image.shape[1]
    transform_matrix = cv2.getRotationMatrix2D(
                (padded_height/2,
                 padded_width/2), # center
                angle, # angle
      1.0) # scale
    rotated_image = cv2.warpAffine(padded_image,
                                   transform_matrix,
                                   (diagonal, diagonal),
                                   flags=cv2.INTER_LANCZOS4)
    
    
    rotated_image[np.where((rotated_image == [0, 0, 0]).all(axis = 2))] = [255, 255, 255]
    return rotated_image
