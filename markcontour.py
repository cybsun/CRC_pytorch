# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:56:43 2019

@author: Chenyu Sun
"""

from PIL import Image
import cv2 
import numpy as np 
import argparse
import os
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
#import imutils

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--eval_dir', default=r'/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/experiments/base_model/output_imgs_test_WSI_TCGA_other')
    #parser.add_argument('--og_dir',default=r'/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_WSI_TCGA_other/test')
    parser.add_argument('--eval_dir', default=r'C:/CRC project/CRC_pytorch_chenyu/experiments/base_model/output_imgs_mxif_tumorcell_bottom2last')
    parser.add_argument('--og_dir',default=r'C:/CRC project/CRC_pytorch_chenyu/data/tumorcell_640_40x_4class/test')
    parser.add_argument('--nu_dir', default=r'C:/CRC project/CRC_pytorch_chenyu/experiments/base_model/output_imgs_mxif_tumorcell_bottom2last/nuclei')
    return parser.parse_args()


def extract(image_sg, image_og, og_filenames, classlist, nu_filenames):
#    image_og = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)

    #thresholding
    ret,thresh = cv2.threshold(image_sg,180,255,cv2.THRESH_BINARY)
    #morphological transformation
    kernel = np.ones((5,5),np.uint8)
    #class_img = cv2.dilate(class_img,kernel,iterations = 1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #kernel = np.ones((9,9),np.uint8)
    #class_img = cv2.erode(class_img,kernel,iterations = 1)
    
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    kernel = np.ones((5,5),np.float32)/25
    D = cv2.filter2D(D,-1,kernel)

    localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresh)
    #im = Image.fromarray(D).convert('1')
    #im.show(D)
    
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    l = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
 
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(image_sg.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        #cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)

        # find contours - cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3 
        cv2MajorVersion = cv2.__version__.split(".")[0] 
    
        # check for contours on thresh 
        if int(cv2MajorVersion) == 4: 
            ctrs, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        else: 
            im2, ctrs, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        for i, ctr in enumerate(sorted_ctrs):
            #get centroid
            M = cv2.moments(ctr)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #print(cX, cY)
            else:
                continue
            
            a = int(64/2)
            # Getting ROI 
            roi = image_og[(cY-a):(cY+a), (cX-a):(cX+a), :] 
            if roi.shape != (a*2,a*2,3):
                continue
            l.append([cX,cY,np.asarray(ctr)])
    print('l=',len(l))
        
    s = []
    for j, name in enumerate(nu_filenames): 
        cX_nu = int(name[-11:-8])
        cY_nu = int(name[-7:-4])
        s.append([cX_nu,cY_nu,classlist[j]])
            
    
    for m in range(len(l)):
        for n in range(len(s)):
            if l[m][0] == s[n][0] and l[m][1] == s[n][1]:
                    
                if int(s[n][2]) == 1:
                # draw the contour and center of the shape on the image
                    cv2.drawContours(image_og, [l[m][2]], -1, (0, 255, 255), 2)
                    
                elif int(s[n][2]) == 0:
                    cv2.drawContours(image_og, [l[m][2]], -1, (255, 0, 0), 2)
                
            
#    labels = labels + 50
#    labels[labels==50] = 0
    im = Image.fromarray(image_og).convert('RGB')
#    im.show(image_og)
    savepath = os.path.join(os.path.join('C:/CRC project/CRC_pytorch_chenyu/experiments/base_model/output_imgs_mxif_tumorcell_bottom2last/', 'marked1'), ('%s.png' %(og_filenames)))
    cv2.imwrite(savepath, image_og)
        
if __name__ == '__main__':

    args=get_args()
    eval_filenames=[f for f in sorted(os.listdir(args.eval_dir)) if f.endswith('_class_1.png')]
    #eval_filenames=[f for f in sorted(os.listdir(args.eval_dir)) if f.endswith('_labels2.png')]
    og_filenames=[f for f in sorted(os.listdir(args.og_dir)) if not (f.endswith('_labels.png') or f.endswith('.xcf') or f.endswith('_labels1.png') or f.endswith('_labels2.png'))]
    #og_filenames=og_filenames[0]
    nu_filenames=[f for f in sorted(os.listdir(args.nu_dir)) if f.endswith('.png')]

    with open('C:/CRC project/CRC_pytorch_chenyu/src/coarseclassifier_40x_64_jitter_e80_more_f3_pred_tumorcell_test_nuclei.txt', 'r') as f:
        a = f.readlines()
    count = 0
    for i in range(len(og_filenames)):
        #fnameroot=eval_filenames[i].replace('_labels2.png','')
        #sub_filenames=[f for f in og_filenames if f.startswith(fnameroot)]
        fnameroot = og_filenames[i].replace('.png','')
        image_og = cv2.imread(os.path.join(args.og_dir,og_filenames[i]))
        image_sg = cv2.imread(os.path.join(args.eval_dir,eval_filenames[i]))  #depends on which class number is nucluei
        #grayscale 
        image_sg = cv2.cvtColor(image_sg, cv2.COLOR_BGR2GRAY) 
        nunu_filenames=[f for f in sorted(nu_filenames) if f.startswith(fnameroot)]
        classlist = a[count:count+len(nunu_filenames)]
        count = count + len(nunu_filenames)
        #print(img_sg_filenames)
        print('count=',len(nunu_filenames))
        #print(image_sg)
        
        extract(image_sg, image_og, fnameroot, classlist, nunu_filenames)