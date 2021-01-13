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

# Robert's Operator
def Robert(image,width,height):
    k1 = np.array([ [-1, 0], [0, 1] ])
    k2 = np.array([ [0, -1], [1, 0] ])
    Gx = np.zeros((width-1,height-1),np.int)
    Gy = np.zeros((width-1,height-1),np.int)
    nwidth = width-1
    nheight = height-1
    for i in range(nwidth):
        for j in range(nheight):
            Gx[i, j] = convolution(image[i:i+2,j:j+2],k1,2,2)
            Gy[i, j] = convolution(image[i:i+2,j:j+2],k2,2,2)
    return np.sqrt((Gx ** 2) + (Gy ** 2))

# Prewitt's Edge Detector
def Prewitt(image,width,height):
    k2 = np.array([ [-1, 0, 1], [-1, 0, 1], [-1, 0, 1] ])
    k1 = np.array([ [-1, -1, -1], [0, 0, 0], [1, 1, 1] ])
    Gx = np.zeros((width-2,height-2),np.int)
    Gy = np.zeros((width-2,height-2),np.int)
    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            Gx[i, j] = convolution(image[i:i+3,j:j+3],k1,3,3)
            Gy[i, j] = convolution(image[i:i+3,j:j+3],k2,3,3)
    return np.sqrt((Gx ** 2) + (Gy ** 2))

# Sobel's Edge Detector
def Sobel(image,width,height):
    k2 = np.array([ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ])
    k1 = np.array([ [-1, -2, -1], [0, 0, 0], [1, 2, 1] ])
    Gx = np.zeros((width-2,height-2),np.int)
    Gy = np.zeros((width-2,height-2),np.int)
    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            Gx[i, j] = convolution(image[i:i+3,j:j+3],k1,3,3)
            Gy[i, j] = convolution(image[i:i+3,j:j+3],k2,3,3)
    return np.sqrt((Gx ** 2) + (Gy ** 2))

# Frei and Chen's Gradient Operator
def FreiAChen(image,width,height):
    k1 = np.array([ [-1, -(2 ** 0.5), -1], [0, 0, 0], [1, (2 ** 0.5), 1] ])
    k2 = np.array([ [-1, 0, 1], [-(2 ** 0.5), 0, (2 ** 0.5)], [-1, 0, 1] ])
    Gx = np.zeros((width-2,height-2),np.int)
    Gy = np.zeros((width-2,height-2),np.int)
    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            Gx[i, j] = convolution(image[i:i+3,j:j+3],k1,3,3)
            Gy[i, j] = convolution(image[i:i+3,j:j+3],k2,3,3)
    return np.sqrt((Gx ** 2) + (Gy ** 2))

# Kirsch's Compass Operator
def Kirsch(image,width,height):
    k0 = np.array([ [-3, -3, 5],[-3, 0, 5],[-3, -3, 5] ])
    k1 = np.array([ [-3, 5, 5],[-3, 0, 5],[-3, -3, -3] ])
    k2 = np.array([ [5, 5, 5],[-3, 0, -3],[-3, -3, -3] ])
    k3 = np.array([ [5, 5, -3],[5, 0, -3],[-3, -3, -3] ])
    k4 = np.array([ [5, -3, -3],[5, 0, -3],[5, -3, -3] ])
    k5 = np.array([ [-3, -3, -3],[5, 0, -3],[5, 5, -3] ])
    k6 = np.array([ [-3, -3, -3],[-3, 0, -3],[5, 5, 5] ])
    k7 = np.array([ [-3, -3, -3],[-3, 0, 5],[-3, 5, 5] ])
    
    img_result = np.zeros((width-2,height-2),np.int)
    candidate = [k0,k1,k2,k3,k4,k5,k6,k7]

    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            candidate_result = []
            for c in candidate:
                candidate_result.append(convolution(image[i:i+3,j:j+3],c,3,3))
            img_result[i,j] = np.max(candidate_result)
    return img_result

# Robinson's Compass Operator
def Robinson(image,width,height):
    k0 = np.array([ [-1, 0, 1],[-2, 0, 2],[-1, 0, 1] ])
    k1 = np.array([ [0, 1, 2],[-1, 0, 1],[-2, -1, 0] ])
    k2 = np.array([ [1, 2, 1],[0, 0, 0],[-1, -2, -1] ])
    k3 = np.array([ [2, 1, 0],[1, 0, -1],[0, -1, -2] ])
    k4 = np.array([ [1, 0, -1],[2, 0, -2],[1, 0, -1] ])
    k5 = np.array([ [0, -1, -2],[1, 0, -1],[2, 1, 0] ])
    k6 = np.array([ [-1, -2, -1],[0, 0, 0],[1, 2, 1] ])
    k7 = np.array([ [-2, -1, 0],[-1, 0, 1],[0, 1, 2] ])
    
    img_result = np.zeros((width-2,height-2),np.int)
    candidate = [k0,k1,k2,k3,k4,k5,k6,k7]

    nwidth = width-2
    nheight = height-2
    for i in range(nwidth):
        for j in range(nheight):
            candidate_result = []
            for c in candidate:
                candidate_result.append(convolution(image[i:i+3,j:j+3],c,3,3))
            img_result[i,j] = np.max(candidate_result)

    return img_result

# Nevatia-Babu 5x5 Operator
def Nevatia(image,width,height):
    k0 = np.array([
        [100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [0, 0, 0, 0, 0],
        [-100, -100, -100, -100, -100],
        [-100, -100, -100, -100, -100],
    ])
    k1 = np.array([
        [100, 100, 100, 100, 100],
        [100, 100, 100, 78, -32],
        [100, 92, 0, -92, -100],
        [32, -78, -100, -100, -100],
        [-100, -100, -100, -100, -100]
    ])
    k2 = np.array([
        [100, 100, 100, 32, -100],
        [100, 100, 92, -78, -100],
        [100, 100, 0, -100, -100],
        [100, 78, -92, -100, -100],
        [100, -32, -100, -100, -100]
    ])
    k3 = np.array([
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100]
    ])
    k4 = np.array([
        [-100, 32, 100, 100, 100],
        [-100, -78, 92, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, -92, 78, 100],
        [-100, -100, -100, -32, 100]
    ])
    k5 = np.array([
        [100, 100, 100, 100, 100],
        [-32, 78, 100, 100, 100],
        [-100, -92, 0, 92, 100],
        [-100, -100, -100, -78, 32],
        [-100, -100, -100, -100, -100]
    ])

    img_result = np.zeros((width-4,height-4),np.int)
    candidate = [k0,k1,k2,k3,k4,k5]

    nwidth = width-4
    nheight = height-4
    for i in range(nwidth):
        for j in range(nheight):
            candidate_result = []
            for c in candidate:
                candidate_result.append(convolution(image[i:i+5,j:j+5],c,5,5))
            img_result[i,j] = np.max(candidate_result)

    return img_result

# Main Function
'''
Usage: python3 hw9.py -p lena.bmp -t a
'''
def main():
    img,width,height = read_img(args.p)
    thresholda = 15
    thresholdb = 24
    thresholdc = 38
    thresholdd = 30
    thresholde = 135
    thresholdf = 43
    thresholdg = 12500

    if(args.t=='a'):
        img1 = (Robert(img,width,height)<=thresholda) * 255
        save_image('Robert_Operator',img1)
    elif(args.t=='b'):
        img2 = (Prewitt(img,width,height)<=thresholdb) * 255
        save_image('Prewitt_Operator',img2)
    elif(args.t=='c'):
        img3 = (Sobel(img,width,height)<=thresholdc) * 255
        save_image('Sobel_Operator',img3)
    elif(args.t=='d'):
        img4 = (FreiAChen(img,width,height)<=thresholdd) * 255
        save_image('Frei_and_Chen_Operator',img4)
    elif(args.t=='e'):
        img5 = (Kirsch(img,width,height)<=thresholde) * 255
        save_image('Kirsch_Operator',img5)
    elif(args.t=='f'):
        img6 = (Robinson(img,width,height)<=thresholdf) * 255
        save_image('Robinson_Operator',img6)
    elif(args.t=='g'):
        img7 = (Nevatia(img,width,height)<=thresholdg) * 255
        save_image('Nevatia_Operator',img7)
    elif(args.t=='all'):
        img1 = (Robert(img,width,height)<=thresholda) * 255
        save_image('Robert_Operator',img1)
        img2 = (Prewitt(img,width,height)<=thresholdb) * 255
        save_image('Prewitt_Operator',img2)
        img3 = (Sobel(img,width,height)<=thresholdc) * 255
        save_image('Sobel_Operator',img3)
        img4 = (FreiAChen(img,width,height)<=thresholdd) * 255
        save_image('Frei_and_Chen_Operator',img4)
        img5 = (Kirsch(img,width,height)<=thresholde) * 255
        save_image('Kirsch_Operator',img5)
        img6 = (Robinson(img,width,height)<=thresholdf) * 255
        save_image('Robinson_Operator',img6)
        img7 = (Nevatia(img,width,height)<=thresholdg) * 255
        save_image('Nevatia_Operator',img7)

if __name__ == "__main__":
    main()