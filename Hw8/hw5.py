import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')


def dilation(image, Kernel,):
    ra, ca = image.shape 
    res = np.zeros(image.shape, dtype = 'int32')
    for ai in range(ra):
        for aj in range(ca):
                
                max_value = 0
                for b_each in Kernel:
                    new_i, new_j = b_each
                    if( (ai + new_i >= 0) and (ai + new_i < ra) and (aj + new_j >= 0) and (aj + new_j < ca) ): 
                        max_value = max(max_value, image[ai + new_i, aj + new_j])
                
                res[ai, aj] = max_value         
    return res 

def erosion(image, Kernel):
    ra, ca = image.shape
    res = np.zeros(image.shape, dtype = 'int32')
    
    for ai in range(ra):
        for aj in range(ca):
                
                min_value = 999999999999999
                for b_each in Kernel:
                    new_i, new_j = b_each
                    if( (ai + new_i >= 0) and (ai + new_i < ra)  and (aj + new_j >= 0) and (aj + new_j < ca) ): 
                        min_value = min(min_value, image[ai + new_i, aj + new_j])
                    
                res[ai, aj] = min_value
    
    return res 

def opening_img(image,Kernel,width,height):
    return dilation(erosion(image, Kernel), Kernel)

def closing_img(image,Kernel,width,height):
    return erosion(dilation(image, Kernel), Kernel)