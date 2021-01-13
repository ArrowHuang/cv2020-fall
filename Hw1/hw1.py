import cv2
import argparse
import numpy as np 

parser = argparse.ArgumentParser(description='How to use this code')
parser.add_argument('-p',default='lena.bmp',type=str,help='The path of image') #image path
parser.add_argument('-t',default='all',type=str,help='The type of user want') #which one user would like to get
args = parser.parse_args()

# Read Image
def read_img(path):
    img = cv2.imread(path)
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

# (a) Upside-down
def upside2down(image,width,height):
    img_result = image.copy()
    for i in range(width):
        img_result[i,:] = image[width-i-1,:]
    save_image('Upside-down',img_result)
    return img_result

# (b) Right-side-left
def right2left(image,width,height):
    img_result = image.copy()
    for i in range(height):
        img_result[:,i] = image[:,height-i-1]
    save_image('Right-side-left',img_result)
    return img_result

# (c) Diagonally-flip
def diagonally_flip(image,width,height):
    img_result = image.copy()
    for i in range(width):
        for j in range(height):
            img_result[j,i] = image[i,j]
    save_image('Diagonally-flip',img_result)
    return img_result

# Main Function
'''
Usage: python hw1.py -p lena.bmp -t all
'''
def main():
    img,width,height = read_img(args.p)
    if(args.t=='1'):
        img1 = upside2down(img,width,height)
    elif(args.t=='2'):
        img2 = right2left(img,width,height)
    elif(args.t=='3'):
        img3 = diagonally_flip(img,width,height)
    elif(args.t=='all'):
        img1 = upside2down(img,width,height)
        img2 = right2left(img,width,height)
        img3 = diagonally_flip(img,width,height)

if __name__ == "__main__":
    main()