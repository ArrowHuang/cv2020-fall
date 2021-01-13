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
                x_list.append( (i,j) )
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

# (a) A binary image (threshold at 128)
def binary_image(image,width,height,T):
    img_result = image.copy()
    for i in range(width):
        for j in range(height):
            if(img_result[i,j]>=128):
                img_result[i,j] = 255
            else:
                img_result[i,j] = 0
    if(T==True):
        save_image('Binary_image',img_result)
    return img_result

# (b) A histogram
def img_histogram(image,width,height):
    Y = np.zeros(256)
    X = [i for i in range(256)]
    for i in range(width):
        for j in range(height):
            Y[image[i,j]] += 1 
    plt.bar(X,Y)
    plt.savefig('Histogram_image.jpg')
    

# (c) connected components(regions with + at centroid, bounding box)
def connect_compoent(image,width,height):
    img1 = binary_image(image,width,height,False)
    img_result = img1.copy()
    label_num = 1
    label_dic = {}

    # First-Pass
    for i in range(width):
        for j in range(height):
            if(img_result[i,j]==255):
                if(i==0):
                    if(j==0 or img_result[i,j-1]==0):
                        img_result[i,j] = label_num
                        label_num = label_num + 1
                    else:
                        img_result[i,j] = img_result[i,j-1]
                else:
                    if(img_result[i-1,j]==0):
                        if(j==0 or img_result[i,j-1]==0):
                            img_result[i,j] = label_num
                            label_num = label_num + 1
                        else:
                            img_result[i,j] = img_result[i,j-1]
                    else:
                        if(j==0 or img_result[i,j-1]==0):
                            img_result[i,j] = img_result[i-1,j]
                        else:
                            if(img_result[i-1,j]<=img_result[i,j-1]):
                                img_result[i,j] = img_result[i-1,j]
                            else:
                                img_result[i,j] = img_result[i,j-1]

                            if(img_result[i-1,j]!=img_result[i,j-1]):
                                if(len(label_dic.keys())==0):
                                    cand_list = [ img_result[i,j-1],img_result[i-1,j] ]
                                    r_list = [np.max(cand_list)]
                                    label_dic[np.min(cand_list)] = r_list
                                else:
                                    cand_list = [ img_result[i,j-1],img_result[i-1,j] ]
                                    if(np.min(cand_list) in label_dic.keys()):
                                        label_dic[np.min(cand_list)] = list(set(label_dic[np.min(cand_list)] + [np.max(cand_list)]))
                                    else:
                                        cand_list = [ img_result[i,j-1],img_result[i-1,j] ]
                                        r_list = [np.max(cand_list)]
                                        label_dic[np.min(cand_list)] = r_list

    # print(label_dic)                    
    sorted_dic = {}
    for i in sorted(label_dic):
        sorted_dic[i] = label_dic[i]
                  
    # # Second-Pass
    count_dic = {}
    for i in range(width):
        for j in range(height):
            for k in sorted_dic.keys():
                if(img_result[i,j] in sorted_dic[k]):
                    img_result[i,j] = k
                    if(img_result[i,j] not in count_dic.keys()):
                        count_dic[img_result[i,j]] = 1
                    else:
                        count_dic[img_result[i,j]] = count_dic[img_result[i,j]] + 1
                    break

    for k in count_dic.keys():
        if(count_dic[k]>=500):
            p1,p2 = get_xy(img_result,width,height,k)
            # print(p1,p2)
            draw = cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
            draw = cv2.circle(draw, (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)), 5,(0, 0, 255),-1)
    save_image('connect_image',draw)


# (c) connected components(regions with + at centroid, bounding box)
def connect_compoent_v2(image):
    img = cv2.imread('lena.bmp')
    img_seg = (image==255)-1
    queue, border = [], []
    idx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_seg[i,j]==0:
                idx += 1
                queue.append((i,j))
                xmin, ymin = img_seg.shape[0], img_seg.shape[1]
                xmax, ymax = 0, 0
                xsum, ysum = 0, 0
                count = 0
                
                while len(queue)!=0:
                    x, y = queue.pop()
                    xmin = x if x<xmin else xmin
                    ymin = y if y<ymin else ymin
                    xmax = x if x>xmax else xmax
                    ymax = y if y>ymax else ymax
                    xsum += x 
                    ysum += y
                    count += 1
                    img_seg[x, y] = idx
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        if (x+dx >= 0) and (x+dx < img.shape[0]) and\
                        (y+dy >= 0) and (y+dy < img.shape[1]) and\
                        img_seg[x+dx, y+dy]==0:
                        
                            queue.append((x+dx, y+dy))
                                
                if count>=500:
                    border.append((xmin, xmax, ymin, ymax, 
                                xsum//count, ysum//count))
    
    for c in border:
        print(c)
        draw = cv2.rectangle(img, (c[2],c[0]), (c[3],c[1]), (0, 0, 255), 2)
        draw = cv2.circle(draw, (int((c[2]+c[3])/2), int((c[0]+c[1])/2)), 5, (0, 0, 255), -1)
    # print(coord)
    save_image('connect_image',draw)

# Main Function
'''
Usage: python3 hw2.py -p lena.bmp -t 3
'''
def main():
    img,width,height = read_img(args.p)
    if(args.t=='1'):
        img1 = binary_image(img,width,height,True)
    elif(args.t=='2'):
        img2 = img_histogram(img,width,height)
    elif(args.t=='3'):
        bimg = binary_image(img,width,height,False)
        connect_compoent_v2(bimg)
    elif(args.t=='all'):
        img1 = binary_image(img,width,height,True)
        img2 = img_histogram(img,width,height)
        bimg = binary_image(img,width,height,False)
        connect_compoent_v2(bimg)


if __name__ == "__main__":
    main()