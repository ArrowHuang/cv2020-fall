import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='How to use this code')
parser.add_argument('-p',default='lena.bmp',type=str,help='The path of image') #image path
parser.add_argument('-t',default='all',type=str,help='The type of user want') #which one user would like to get
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

# Get the maxinum of dilation
def maxnum_dilation(image,kernel,i,j,width,height):
    max_num = 0
    for k in kernel:
        new_i = i + k[0]
        new_j = j + k[1]
        if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
            max_num = max(max_num,image[new_i,new_j])
    return max_num

# (a) Dilation image 往外膨脹
def dilation_img(image,kernel,width,height):
    img_result = image.copy()
    for i in range(width):
        for j in range(height):
            if(image[i,j]!=0):
                max_value = maxnum_dilation(image,kernel,i,j,width,height)
                for k in kernel:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
                        img_result[new_i,new_j] = max_value
    return img_result

# Get the mininum of erosion
def mininum_erosion(image,kernel,i,j,width,height):
    correct_num = 0
    min_num = 999999
    for k in kernel:
        new_i = i + k[0]
        new_j = j + k[1]
        if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
            if(image[new_i,new_j]!=0):
                correct_num = correct_num + 1
                min_num = min(min_num,image[new_i,new_j])
            else:
                break
        else:
            break
    return min_num

# (b) Erosion image 往內侵蝕
def erosion_img(image,kernel,width,height):
    img_result = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            if(image[i,j]!=0):
                min_value = mininum_erosion(image,kernel,i,j,width,height)
                for k in kernel:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
                        if(image[new_i,new_j]!=0):
                            img_result[new_i,new_j] = min_value
                        else:
                            break
                    else:
                        break
    return img_result

# (c) Opening image 先Erosion再Dilation
def opening_img(image,kernel,width,height):
    e_img = erosion_img(image,kernel,width,height)
    img_result = dilation_img(e_img,kernel,width,height)
    return img_result

# (d) Closing image 先Dilation再Erosion
def closing_img(image,kernel,width,height):
    d_img = dilation_img(image,kernel,width,height)
    img_result = erosion_img(d_img,kernel,width,height)
    return img_result

kernel = [
         [-2,-1],[-2,0],[-2,1],
 [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
 [0,-2], [0,-1], [0,0], [0,1], [0,2],
 [1,-2], [1,-1], [1,0], [1,1], [1,2],
         [2,-1],[2,0],[2,1],
]

# Main Function
'''
Usage: python hw5.py -p lena.bmp -t 1
'''
def main():
    img,width,height = read_img(args.p)
    if(args.t=='1'):
        img1 = dilation_img(img,kernel,width,height)
        save_image('Dilation_img',img1)
    elif(args.t=='2'):
        img2 = erosion_img(img,kernel,width,height)
        save_image('Erosion_img',img2)
    elif(args.t=='3'):
        img3 = opening_img(img,kernel,width,height)
        save_image('Opening_img',img3)
    elif(args.t=='4'):
        img4 = closing_img(img,kernel,width,height)
        save_image('Closing_img',img4)
    elif(args.t=='all'):
        img1 = dilation_img(img,kernel,width,height)
        save_image('Dilation_img',img1)

        img2 = erosion_img(img,kernel,width,height)
        save_image('Erosion_img',img2)

        img3 = opening_img(img,kernel,width,height)
        save_image('Opening_img',img3)

        img4 = closing_img(img,kernel,width,height)
        save_image('Closing_img',img4)

if __name__ == "__main__":
    main()