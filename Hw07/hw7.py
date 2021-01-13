import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='How to use this code')
parser.add_argument('-p',default='lena.bmp',type=str,help='The path of image') #image path
args = parser.parse_args()

# Get Region
def get_xy(image,width,height,label):
    x_list = []
    for i in range(width):
        for j in range(height):
            if(image[i,j]==label):
                x_list.append( (j,i) )
    return x_list[0],x_list[-1]

# Read Image
def read_img(path):
    img = cv2.imread(path,0)
    width = img.shape[0]
    height = img.shape[1]
    return img,width,height

# Show Image
def show_img(image):
    cv2.imshow('My Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save Image
def save_image(path_name,image):
    cv2.imwrite(path_name+'.jpg', image)

# Binary image (threshold at 128)
def binary_image(image,width,height):
    img_result = image.copy()
    for i in range(width):
        for j in range(height):
            if(img_result[i,j]>=128):
                img_result[i,j] = 255
            else:
                img_result[i,j] = 0
    return img_result

# Downsampling image (64x64)
def downsample_image(image,height,width):
    img_downsample = np.zeros((height,width),np.int)
    for i in range(height):
        for j in range(width):
            img_downsample[i,j] = image[i*8,j*8]
    return img_downsample

kernel = [
    [-1,-1],[-1,0],[-1,1],
    [0,-1],[0,0],[0,1],
    [1,-1],[1,0],[1,1]
]

def h(b,c,d,e):
    if(b == c):
        if(d == b and e == b):
            return 'r'
        else:
            return 'q'
    return 's'


def yokoi_count(tmp_kernel):
    h1 = h(tmp_kernel[1,1], tmp_kernel[1,2], tmp_kernel[0,2], tmp_kernel[0,1])
    h2 = h(tmp_kernel[1,1], tmp_kernel[0,1], tmp_kernel[0,0], tmp_kernel[1,0])
    h3 = h(tmp_kernel[1,1], tmp_kernel[1,0], tmp_kernel[2,0], tmp_kernel[2,1])
    h4 = h(tmp_kernel[1,1], tmp_kernel[2,1], tmp_kernel[2,2], tmp_kernel[1,2])
    
    return [h1,h2,h3,h4]

# Yokoi Connectivity Number
def yokoi_img(image,height,width):
    img_result = image.copy()
    for i in range(width):
        for j in range(height):
            result_list = []
            tmp_kernel = np.zeros((3,3))
            if(image[i,j]==255):
                for k in kernel:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if(new_i>=0 and new_i<=height-1 and new_j>=0 and new_j<=width-1):
                        if(image[new_i,new_j]==255):
                            tmp_kernel[k[0]+1,k[1]+1] = 1   
                result_list = yokoi_count(tmp_kernel)
                if( result_list.count('r')==4 ):
                    img_result[i,j] = int(5)
                else:
                    img_result[i,j] = int(result_list.count('q'))

    # print(img_result)
    return img_result

# Pair Relation Operator
def pair_relation_img(image,height,width):
    img_result = np.zeros((image.shape[0],image.shape[1]),np.str)
    for i in range(height):
        for j in range(width):
            if(image[i,j]>0):
                if(image[i,j]!=1):
                    img_result[i,j] = 'q'
                else:
                    tmp_kernel = np.zeros((3,3))
                    for k in kernel:
                        new_i = i + k[0]
                        new_j = j + k[1]
                        if(new_i>=0 and new_i<=height-1 and new_j>=0 and new_j<=width-1):
                            tmp_kernel[k[0]+1,k[1]+1] = image[new_i,new_j]
                    tmp_list = [tmp_kernel[0,1],tmp_kernel[1,0],tmp_kernel[1,2],tmp_kernel[2,1]]
                    if(tmp_list.count(1)>=1):
                        img_result[i,j] = 'p'

    return img_result

# Main Function
'''
Usage: python3 hw7.py -p lena.bmp
'''
def main():
    img,width,height = read_img(args.p)
    img_bi = binary_image(img,width,height)
    img_downsample = downsample_image(img_bi,64,64)
    img_yokoi = yokoi_img(img_downsample,64,64)
    img_pair = pair_relation_img(img_yokoi,64,64)
    
    tmp_img = img_downsample.copy()
    tmp_pair = img_pair.copy()

    for ite in range(10):
        pre_img = tmp_img.copy()
        for i in range(64):
            for j in range(64):
                if(tmp_pair[i,j]=='p'):
                    result_list = []
                    tmp_kernel = np.zeros((3,3))
                    for k in kernel:
                        new_i = i + k[0]
                        new_j = j + k[1]
                        if(new_i>=0 and new_i<=63 and new_j>=0 and new_j<=63):
                            if(tmp_img[new_i,new_j]==255):
                                tmp_kernel[k[0]+1,k[1]+1] = 1   
                        result_list = yokoi_count(tmp_kernel)
                        
                    if( result_list.count('q')==1 ):
                        tmp_img[i,j] = 0
        
        tmp_yoki = yokoi_img(tmp_img,64,64)   
        tmp_pair = pair_relation_img(tmp_yoki,64,64)
        if((pre_img==tmp_img).all()):
            save_image('Thinning_Operator',tmp_img)
            break

if __name__ == "__main__":
    main()