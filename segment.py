import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage

import copy
import sys

# Thresholding and connected component labeling

def connected_component_label(path):    
    # Read the input image and convert to a gray scale image
    inp_img = cv2.imread(path)
    gray_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    #
    # Originally we simply thresholded at half of the maximum value uniformly
    # everywhere. This fixed thresholding was only partially effective.
    # Converting those pixels with values 1-127 to 0 and others to 1
    #img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]

    # Then we switched to adaptive thresholding called here
    img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY,11,2)

    # The images may have to be inverted (black on white vs. white on black)
    # or rescaled in several places
    img = 255 - img
    print('SHAPE', img.shape)
    # Applying connected components analysis, i.e. extracting islands of pixels
    # above the threshold (think land above water). Each island will roughly
    # contain a single character of the captcha text, occasionally two characters.
    num_labels, labels = cv2.connectedComponents(img)

    #print(type(labels), labels.shape)

    # This gets rid of small islands (smaller than 80 pixels)
    new_label = 1
    for i in range(np.max(labels)):
        if i == 0:
            continue
        idx = np.where(labels==i)
        num_pixels = len(idx[0])
        if num_pixels > 80:
            # This was used to "spread" the range of colors,
            # but turned out to be unnecessary
            pass
        else:
            labels[idx] = 0

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    # This is only needed to mark each island with different color
    # and visualize progress.
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set background color label to black
    labeled_img[label_hue==0] = 0
    
    # Showing Original Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.axis("off")
    plt.title("Orginal Image")
    plt.show()
    
    # Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()

    # Return labels and gray image to be further processed,
    # and labeled image showing which pixels belong to which character
    return labels, gray_img, labeled_img


# This class allows us to manipulate postage stamp images
# of detected characters (islands of black print).
# It stores a tiny image of a single character, sometimes two
# characters that will be split later. It also stores
# the x, y location and extent of the postage stamp in the input
# captcha image. Stamp objects can also compute their own size,
# "center", and can pretty print themselves.

class Stamp(object):

    def __init__(self, label, idx, img):
        self.label = label
        self.idx = idx
        self.img = img
        self.npix = len(self.idx[0])
        self.xlo = np.min(self.idx[1])
        self.xhi = np.max(self.idx[1])
        self.ylo = np.min(self.idx[0])
        self.yhi = np.max(self.idx[0])
        self.x0 = int(.5*(self.xlo+self.xhi+1))
        self.y0 = int(.5*(self.ylo+self.yhi+1))

    def xsize(self):
        return self.xhi-self.xlo+1

    def ysize(self):
        return self.yhi-self.ylo+1
        
    def __str__(self):
        return (10*'\t%5d') % (self.label,
                               self.npix,
                               self.x0,
                               self.y0,
                               self.xsize(),
                               self.ysize(),
                               self.xlo,
                               self.xhi,
                               self.ylo,
                               self.yhi)

# This extracts postage stamp images of characters
# from the original captcha image based on the labels
# from connected component analysis.
#
# This may be a bit confusing because several things
# are done simultaneously as the algorithm goes through all data.

def extract_stamps(label_pattern):
    # Get pixel locations and pixel count of all detected patterns (islands/characters)
    patterns = []
    labels = set()
    for l in range(np.max(label_pattern)):
        if l == 0:
            continue
        idx = np.where(label_pattern==l)
        num_pixels = len(idx[0])
        patterns.append((l, idx, num_pixels))
        labels.add(l)
    #print(labels)
    #
    print([(p[0], p[2]) for p in patterns])
    # Sort pixel patterns based on their area
    # (number of dark pixels that form the pattern)
    spatterns = sorted(patterns, key=lambda p: p[2], reverse=True)
    #print([(p[0], p[2]) for p in spatterns])
    #
    # Starting from empty list of stamps,
    # extract 5 largest stamps, skipping
    # abnormal cases that were zeroed out.
    stamps = []
    i = 0
    print('NUM Patterns', len(spatterns))
    print('\tlabel npix x0 y0 xsize ysize xlo xhi ylo yhi')
    #
    while len(stamps) < 5 and i < len(spatterns):
        l, idx, npix = spatterns[i]
        idx_y, idx_x = idx
        #print(idx_x, idx_y)
        #print(i)
        if (len(idx_x) < 1 or len(idx_y) < 1):
            i += 1
            continue
        stamp = Stamp(l, idx, None)
        stamp.img = label_pattern[stamp.ylo:stamp.yhi+1,
                                  stamp.xlo:stamp.xhi+1]
        print(stamp)
        #
        stamps.append(stamp)
        #
        i += 1
    #
    # Return 5 largest stamps because we expect 5 characters in our chosen examples.
    # At this point some of them may still be double characters and will be split later.
    return stamps

# This splits stamps that look like doubles
def clean_stamps(stamps, gray_img, labeled_img):
    single_stamps = []
    #median_xsize = sorted([stamp.xsize() for stamp in stamps])[len(stamps)//2]
    median_xsize = np.median([stamp.xsize() for stamp in stamps])
    for stamp in stamps:
        #print(stamp.npix, stamp.idx)
        # Deep copy needed here to prevent data corruption
        stamp_gray_img = copy.deepcopy(gray_img[stamp.ylo:stamp.yhi+1,
                                                stamp.xlo:stamp.xhi+1])
        stamp_label_img = copy.deepcopy(labeled_img[stamp.ylo:stamp.yhi+1,
                                                    stamp.xlo:stamp.xhi+1])
        # Showing Original Stamp Image and Label Image
        #print('PIXELS:', stamp_gray_img)
        #print('LABELS:', stamp_label_img)        
        #plt.imshow(cv2.cvtColor(stamp_gray_img, cv2.COLOR_BGR2RGB))
        #plt.axis("off")
        #plt.title("Orginal Stamp Image")
        #plt.show()
        #plt.imshow(cv2.cvtColor(stamp_label_img, cv2.COLOR_BGR2RGB))
        #plt.axis("off")
        #plt.title("Orginal Stamp Image")
        #plt.show()
        #
        # Stamps may overlap slightly due to distorted print in captcha images.
        # In this stamp we erase (set to white) any pixels that belong to other characters.
        erase_idx = np.where(stamp.img!=stamp.label)
        #print(stamp.label, stamp.img)
        #print(erase_idx)
        stamp_gray_img[erase_idx] = 255
        stamp_label_img[erase_idx] = [0,0,0]
        #print(stamp_label_img)
        # Showing Original Stamp Image
        #print('STAMP SIZE:', stamp.xsize(), stamp.ysize())
        #print('PIXELS:', stamp_gray_img)
        #plt.imshow(cv2.cvtColor(stamp_gray_img, cv2.COLOR_BGR2RGB))
        #plt.axis("off")
        #plt.title("Orginal Stamp Image")
        #plt.show()
        # TODO: split stamps unusually large in x (they are doubles or even tripples)

        # If the current stamp is wide in x compared to median size,
        # we split it near middle at the lowest point of the black
        # pixel histogram (black pixel count in each column as a function of x).
        # Both stamps resulting from the split are added to the list of single stamps.
        #
        # If the stamp is "normal" size, it is most likely a single character,
        # in which case we simply add it to single stamps.
        #
        # Stamps are added with the x location of their middle point
        # so that later we can sort from left to right to print actual captcha text.
        #
        if stamp.xsize() > 1.33*median_xsize:
            # Split the stamp
            stamp_hist = np.sum(stamp_gray_img<255, 0)
            #stamp_hist = np.sum(stamp_gray_img, 0)
            tol = 0.2
            idx_size = stamp_hist.shape[0]
            idx_mid = int(idx_size*0.5)
            idx_lo  = int(idx_size*0.5*(1.-tol))
            idx_hi  = int(idx_size*0.5*(1.+tol))
            #print(idx_size, idx_mid, idx_lo, idx_hi)
            #print(stamp_hist[idx_lo:idx_hi])
            idx_split = idx_lo + np.argmin(stamp_hist[idx_lo:idx_hi])
            #print('SPLIT:', idx_split)
            stamp_gray_img1 = stamp_gray_img[:,:idx_split]
            stamp_gray_img2 = stamp_gray_img[:,idx_split:]
            #
            check_img = np.concatenate((stamp_gray_img1, stamp_gray_img2), axis=1)
            check_img = np.concatenate((check_img, stamp_gray_img), axis=0)
            plt.imshow(cv2.cvtColor(check_img, cv2.COLOR_GRAY2RGB))
            #plt.imshow(cv2.cvtColor(stamp_gray_img1, cv2.COLOR_GRAY2RGB))
            plt.axis("off")
            plt.title("Check Double Stamp Image")
            plt.show()
            #plt.imshow(cv2.cvtColor(stamp_gray_img2, cv2.COLOR_GRAY2RGB))
            #plt.axis("off")
            #plt.title("Check Double Stamp Image")
            #plt.show()
            #
            single_stamps.append((stamp_gray_img1, stamp.x0-stamp_gray_img1.shape[1]//2))
            single_stamps.append((stamp_gray_img2, stamp.x0+stamp_gray_img2.shape[1]//2))            
            #print(stamp_hist)
            plt.plot(stamp_hist)
            plt.show()
            print('Stamp shape:', stamp_gray_img1.shape)
            print('Stamp shape:', stamp_gray_img2.shape)            
        else:
            single_stamps.append((stamp_gray_img, stamp.x0))
            print('Stamp shape:', stamp_gray_img.shape)
            plt.imshow(cv2.cvtColor(stamp_gray_img, cv2.COLOR_GRAY2RGB))
            plt.axis("off")
            plt.title("Check Single Stamp Image")
            plt.show()
    # TODO: resize stamps to the same size
    #
    # Sort by increasing x of the middle of the stamp,
    # which is from left to right.
    single_stamps = sorted(single_stamps, key=lambda s: s[1])
    print([s[1] for s in single_stamps])
    # Return single stamps sorted from left to right
    return single_stamps

# This resizes the final single character stamps
# to the same size as expected by the neural net model (28x28)
# and also scales/inverts the final postage stamp image
# to be gray scale valued between 0.0 and 1.0 (black on white)
# Resizing is done using two-dimensional interpolation.
def rescale_stamp(stamp_img, shape=[24,24], margin=[2,2]):
    ny, nx = stamp_img.shape[0], stamp_img.shape[1]
    rescaled_stamp_img = scipy.ndimage.zoom(stamp_img/255.,
                                            zoom=(float(shape[0])/ny, float(shape[1])/nx),
                                            output=np.float32,
                                            cval=1.0)
    #
    rescaled_stamp_img[np.where(rescaled_stamp_img>1.0)] = 1.0
    rescaled_stamp_img[np.where(rescaled_stamp_img<0.0)] = 0.0
    result = np.zeros(shape=(shape[0]+2*margin[0],
                             shape[1]+2*margin[1]))
    result[margin[0]:-margin[0],
           margin[1]:-margin[1]] = 1.0 - rescaled_stamp_img
    return result


#################################################################

# If this program is invoked incorrectly, print a hint and exit
if len(sys.argv) != 2:
    print('usage: python %s image_filename (e.g. fc001.jpg)' % sys.argv[0])
    sys.exit(1)
    
fname = sys.argv[1]    

####################################
# Apply all preprocessing steps
####################################

# Read captcha image and perform connected component labeling
labels, img, labeled_img = connected_component_label(fname)

# Cut out postage stamp images around single characters
stamps = extract_stamps(labels)

# Split double characters that
clean_stamps = clean_stamps(stamps, img, labeled_img)

########################################################################
# Load the deep neural network model
#
# This was created and trained by another program mnist_convnet.py
# The trained model was stored in subfolder mnist_convnet.model
########################################################################

from tensorflow import keras
model = keras.models.load_model('mnist_convnet.model')
#
digits = '0123456789'

# Begin to assemble captcha text starting from an empty string
captcha_prediction = ''

for stamp_img, x_location in clean_stamps:
    # Resize and rescale gray levels so that we can present
    # the data to the deep neural net model as expected.
    resized_stamp_img = rescale_stamp(stamp_img)
    print ('X location/size:', x_location, stamp_img.shape, resized_stamp_img.shape)
    #plt.imshow(cv2.cvtColor(stamp_img, cv2.COLOR_GRAY2RGB))
    #plt.axis("off")
    #plt.title("Clean Stamp Image")
    #plt.show()
    #plt.imshow(cv2.cvtColor(resized_stamp_img, cv2.COLOR_GRAY2RGB))
    #plt.axis("off")
    #plt.title("Clean Stamp Image")
    #plt.show()
    inp_data = np.expand_dims(resized_stamp_img, -1)
    ########    ########    ########    ########    ########
    # The point of all this laborious preprocessing was to
    # get to this place where we finally use the neural net
    # to recognize single characters
    ########    ########    ########    ########    ########
    out_data = model.predict(np.array([inp_data]))
    out_class = model.predict_classes(np.array([inp_data]))
    #digit_idx = np.argmax(out_data)
    #out_digit = digits[digit_idx]
    print('PREDICTION:', out_class, out_data)
    plt.imshow(1.-resized_stamp_img, cmap='gray')
    plt.show()
    # Assemble captcha text one character at a time
    captcha_prediction += str(out_class[0])
    #

# Print the final result, limiting to the expected 5 characters,
# because splitting doubles may have created more than 5.
# If there are fewer than 5, we print them all.
print('##################')
print('Final result:', captcha_prediction[:min(len(captcha_prediction),5)])
