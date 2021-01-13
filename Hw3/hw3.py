import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
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


# (a) Original image and its histogram
def img_histogram(image,width,height):
    Y = np.zeros(256)
    X = [i for i in range(256)]
    for i in range(width):
        for j in range(height):
            Y[image[i,j]] += 1 
    plt.bar(X,Y)
    plt.savefig('Histogram_image.jpg')


# (b) Image with intensity divided by 3 and its histogram
def img_divide3(image,width,height):
    img = image.copy()
    Y = np.zeros(256)
    X = [i for i in range(256)]
    for i in range(width):
        for j in range(height):
            img[i,j] = int(image[i,j]/3)
            Y[img[i,j]] += 1 
    plt.clf()
    plt.cla()
    plt.bar(X,Y)
    plt.savefig('Histogram_image_Divide3.jpg')
    return img

# (c) Image after applying histogram equalization to (b) and its histogram
def histogram_equalization(image,width,height):
    img = image.copy()
    Y = np.zeros(256)
    Y_2 = np.zeros(256)
    Y_h = np.zeros(256)
    X = [i for i in range(256)]
    for i in range(width):
        for j in range(height):
            img[i,j] = int(image[i,j]/3)
            Y[img[i,j]] += 1 
    
    Y_2[0] = Y[0]
    for i in range(1,256):
        Y_2[i] = Y_2[i-1] + Y[i]
    Y_2 = 255*(Y_2/(width*height))

    for i in range(width):
        for j in range(height):
            img[i,j] = Y_2[img[i,j]]
            Y_h[img[i,j]] += 1
    
    plt.clf()
    plt.cla()
    plt.bar(X,Y_h)
    plt.savefig('Histogram_histogram_equalization.jpg')
    return img

# Main Function
'''
Usage: python hw3.py -p lena.bmp -t 1
'''
def main():
    img,width,height = read_img(args.p)
    if(args.t=='1'):
        img1 = img_histogram(img,width,height)
        save_image('original_img',img)
    elif(args.t=='2'):
        img2 = img_divide3(img,width,height)
        save_image('img_divide_three',img2)
    elif(args.t=='3'):
        img3 = histogram_equalization(img,width,height)
        save_image('histogram_equalization',img3)
    elif(args.t=='all'):
        img1 = img_histogram(img,width,height)
        save_image('original_img',img)

        img2 = img_divide3(img,width,height)
        save_image('img_divide_three',img2)

        img3 = histogram_equalization(img,width,height)
        save_image('histogram_equalization',img3)


if __name__ == "__main__":
    main()