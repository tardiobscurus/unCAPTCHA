import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys

# Thresholding and connected component labeling

def connected_component_label(path):    
    # Getting the input image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # Converting those pixels with values 1-127 to 0 and others to 1
    #img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY,11,2)
    img = 255 - img
    print('SHAPE', img.shape)
    # Applying cv2.connectedComponents() 
    num_labels, labels = cv2.connectedComponents(img)

    #print(type(labels), labels.shape)

    # This gets rid of small islands (smaller than 80 pixels)
    # and also attemps to spread the range of colors
    new_label = 1
    for i in range(np.max(labels)):
        if i == 0:
            continue
        idx = np.where(labels==i)
        num_pixels = len(idx[0])
        if num_pixels > 80:
            labels[idx] = new_label
        else:
            labels[idx] = 0
        new_label += 5

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    
    # Showing Original Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Orginal Image")
    plt.show()
    
    #Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()


if len(sys.argv) != 2:
    print('usage: python %s image_filename (e.g. fc001.jpg)' % sys.argv[0])
    sys.exit(1)
    
fname = sys.argv[1]    

connected_component_label(fname)
