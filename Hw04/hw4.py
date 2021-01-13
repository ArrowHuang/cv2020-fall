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

# (a) Dilation image 往外膨脹
def dilation_img(image,kernel,width,height):
    img_result = image.copy()
    for i in range(width):
        for j in range(height):
            if(image[i,j]==255):
                for k in kernel:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
                        img_result[new_i,new_j] = 255
    return img_result

# (b) Erosion image 往內侵蝕
def erosion_img(image,kernel,width,height):
    img_result = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            correct_num = 0
            if(image[i,j]==255):
                for k in kernel:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
                        if(image[new_i,new_j]==255):
                            correct_num = correct_num + 1
                        else:
                            break
                    else:
                        break
                if(correct_num == len(kernel)):
                    img_result[i,j] = 255
    return img_result

# (b)_v2 Erosion image 往內侵蝕
def erosion_img_v2(image,kernel,width,height):
    img_result = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            correct_num = 0
            for k in kernel:
                new_i = i + k[0]
                new_j = j + k[1]
                if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
                    if(image[new_i,new_j]==255):
                        correct_num = correct_num + 1
                    else:
                        break
                else:
                    break
            if(correct_num == len(kernel)):
                img_result[i,j] = 255
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

# (e) Hit-and-miss transform
def hit_and_miss(image,Jkernel,Kkernel,width,height):
    img1 = erosion_img(image,Jkernel,width,height)
    img2 = erosion_img_v2(255-image,Kkernel,width,height)
    img_result = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            if(img1[i,j]==255 and img2[i,j]==255):
                img_result[i,j] = 255
    #show_img(img_result)
    return(img_result)

kernel = [
         [-2,-1],[-2,0],[-2,1],
 [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
 [0,-2], [0,-1], [0,0], [0,1], [0,2],
 [1,-2], [1,-1], [1,0], [1,1], [1,2],
         [2,-1],[2,0],[2,1],
]

kernel2 = [ [0, -1], [0, 0], [1, 0] ]

kernel3 = [ [-1, 0], [-1, 1], [0, 1] ]

# Main Function
'''
Usage: python hw4.py -p lena.bmp -t all
'''
def main():
    img,width,height = read_img(args.p)
    img_bi = binary_image(img,width,height)
    if(args.t=='1'):
        img1 = dilation_img(img_bi,kernel,width,height)
        save_image('Dilation_img',img1)
    elif(args.t=='2'):
        img2 = erosion_img(img_bi,kernel,width,height)
        save_image('Erosion_img',img2)
    elif(args.t=='3'):
        img3 = opening_img(img_bi,kernel,width,height)
        save_image('Opening_img',img3)
    elif(args.t=='4'):
        img4 = closing_img(img_bi,kernel,width,height)
        save_image('Closing_img',img4)
    elif(args.t=='5'):
        img5 = hit_and_miss(img_bi,kernel2,kernel3,width,height)
        save_image('Hit_and_Miss_img',img5)
    elif(args.t=='all'):
        img1 = dilation_img(img_bi,kernel,width,height)
        save_image('Dilation_img',img1)

        img2 = erosion_img(img_bi,kernel,width,height)
        save_image('Erosion_img',img2)

        img3 = opening_img(img_bi,kernel,width,height)
        save_image('Opening_img',img3)

        img4 = closing_img(img_bi,kernel,width,height)
        save_image('Closing_img',img4)

        img5 = hit_and_miss(img_bi,kernel2,kernel3,width,height)
        save_image('Hit_and_Miss_img',img5)


if __name__ == "__main__":
    main()