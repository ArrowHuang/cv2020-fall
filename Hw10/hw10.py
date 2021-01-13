import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import warnings
import math
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

# Convolution
def convolution(img,kernel,width,height):
    value = 0
    for i in range(width):
        for j in range(height):
            value += (img[i, j] * kernel[width-i-1, height-j-1])
    return value

# Zero Cross Edge
def zero_cross_edge(image):
    kernel = [
        [-1,-1],[-1,0],[-1,1],
        [0,-1],[0,0],[0,1],
        [1,-1],[1,0],[1,1]
    ]
    width,height = image.shape
    img_result = np.zeros((width,height),np.int)
    for i in range(width):
        for j in range(height):
            bl = False
            if(image[i,j]==1):
                for k in kernel:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if(new_i>=0 and new_i<=width-1 and new_j>=0 and new_j<=height-1):
                        if(image[new_i,new_j]==-1):
                            bl = True
                            break
            if(bl==False):
                img_result[i,j] = 255
            else:
                img_result[i,j] = 0
    # print(img_result)
    return img_result

# Laplace Mask1
def Laplace1(image,width,height,threshold):
    k1 = np.array([ [0, 1, 0], [1, -4, 1], [0, 1, 0] ])
    img_result = np.zeros((width-2,height-2),np.int)
    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            if(convolution(image[i:i+3,j:j+3],k1,3,3)>=threshold):
                img_result[i,j] = 1
            elif(convolution(image[i:i+3,j:j+3],k1,3,3)<=-threshold):
                img_result[i,j] = -1
            else:
                img_result[i,j] = 0
    return img_result

# Laplace Mask2
def Laplace2(image,width,height,threshold):
    k1 = np.array([ [1, 1, 1], [1, -8, 1], [1, 1, 1] ])
    k1 = k1 / 3
    img_result = np.zeros((width-2,height-2),np.int)
    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            if(convolution(image[i:i+3,j:j+3],k1,3,3)>=threshold):
                img_result[i,j] = 1
            elif(convolution(image[i:i+3,j:j+3],k1,3,3)<=-threshold):
                img_result[i,j] = -1
            else:
                img_result[i,j] = 0
    return img_result

# Minimum Variance Laplacian
def Minium_Variance_Laplace(image,width,height,threshold):
    k1 = np.array([ [2, -1, 2], [-1, -4, -1], [2, -1, 2] ])
    k1 = k1 / 3
    img_result = np.zeros((width-2,height-2),np.int)
    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            if(convolution(image[i:i+3,j:j+3],k1,3,3)>=threshold):
                img_result[i,j] = 1
            elif(convolution(image[i:i+3,j:j+3],k1,3,3)<=-threshold):
                img_result[i,j] = -1
            else:
                img_result[i,j] = 0
    return img_result

# Laplace of Gaussian
def Laplace_Gaussian(image,width,height,threshold):
    k1 = np.array([ [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0], 
                    [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0], 
                    [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                    [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1], 
                    [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1], 
                    [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2], 
                    [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1], 
                    [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                    [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                    [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0], 
                    [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]  ])
    img_result = np.zeros((width-10,height-10),np.int)
    nwidth = width-10
    nheight = height-10
    for i in range(nwidth):
        for j in range(nheight):
            if(convolution(image[i:i+11,j:j+11],k1,11,11)>=threshold):
                img_result[i,j] = 1
            elif(convolution(image[i:i+11,j:j+11],k1,11,11)<=-threshold):
                img_result[i,j] = -1
            else:
                img_result[i,j] = 0
    return img_result

# Difference of Gaussian
def Difference_Gaussian(image,width,height,threshold):
    k1 = np.array([ [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1], 
                    [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3], 
                    [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                    [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6], 
                    [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7], 
                    [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8], 
                    [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7], 
                    [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                    [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                    [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3], 
                    [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]  ])
    img_result = np.zeros((width-10,height-10),np.int)
    nwidth = width-10
    nheight = height-10
    for i in range(nwidth):
        for j in range(nheight):
            if(convolution(image[i:i+11,j:j+11],k1,11,11)>=threshold):
                img_result[i,j] = 1
            elif(convolution(image[i:i+11,j:j+11],k1,11,11)<=-threshold):
                img_result[i,j] = -1
            else:
                img_result[i,j] = 0
    return img_result


# Main Function
'''
Usage: python3 hw10.py -p lena.bmp -t all
'''
def main():
    img,width,height = read_img(args.p)
    thresholda = 15
    thresholdb = 15
    thresholdc = 15
    thresholdd = 3000
    thresholde = 0

    if(args.t=='a'):
        img1 = zero_cross_edge(Laplace1(img,width,height,thresholda))
        save_image('Laplace_Mask1',img1)
    elif(args.t=='b'):
        img2 = zero_cross_edge(Laplace2(img,width,height,thresholdb))
        save_image('Laplace_Mask2',img2)
    elif(args.t=='c'):
        img3 = zero_cross_edge(Minium_Variance_Laplace(img,width,height,thresholdc))
        save_image('Minium_Variance_Laplace',img3)
    elif(args.t=='d'):
        img4 = zero_cross_edge(Laplace_Gaussian(img,width,height,thresholdd))
        save_image('Laplace_Gaussian',img4)
    elif(args.t=='e'):
        img5 = zero_cross_edge(Difference_Gaussian(img,width,height,thresholde))
        save_image('Difference_Gaussian',img5)
    elif(args.t=='all'):
        img1 = zero_cross_edge(Laplace1(img,width,height,thresholda))
        save_image('Laplace_Mask1',img1)
        img2 = zero_cross_edge(Laplace2(img,width,height,thresholdb))
        save_image('Laplace_Mask2',img2)
        img3 = zero_cross_edge(Minium_Variance_Laplace(img,width,height,thresholdc))
        save_image('Minium_Variance_Laplace',img3)
        img4 = zero_cross_edge(Laplace_Gaussian(img,width,height,thresholdd))
        save_image('Laplace_Gaussian',img4)
        img5 = zero_cross_edge(Difference_Gaussian(img,width,height,thresholde))
        save_image('Difference_Gaussian',img5)

if __name__ == "__main__":
    main()