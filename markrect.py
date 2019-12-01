# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:05:45 2019

@author: Chenyu sun
"""
import cv2 
import numpy as np 
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--eval_dir', default=r'/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_WSI_TCGA_nuclei/nuclei10/nuclei10')
    #parser.add_argument('--og_dir',default=r'/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_WSI_TCGA_other/test')
    parser.add_argument('--eval_dir', default=r'C:/CRC project/CRC_pytorch_chenyu/experiments/base_model/output_imgs_mxif_tumorcell_bottom2last/nuclei')
    parser.add_argument('--og_dir',default=r'C:/CRC project/CRC_pytorch_chenyu/data/tumorcell_640_40x_4class/test')
    return parser.parse_args()


def mark(img_sg_filenames, image_og, fnameroot, classlist):
    

    #img_save_dir = os.path.join('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_WSI_TCGA_nuclei', 'marked')
    img_save_dir = os.path.join('C:/CRC project/CRC_pytorch_chenyu/experiments/base_model/output_imgs_mxif_tumorcell_bottom2last', 'marked1')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    
    
    
    for i, name in enumerate(img_sg_filenames): 
        
        cX = int(name[-11:-8])
        cY = int(name[-7:-4])
        #print(cX, cY)
        #print(classlist[i])
        if int(classlist[i]) == 1:
            #cv2.rectangle(image_og,(cX-32,cY-32),(cX+32,cY+32),(0,0,255),3)
            #The parameters here are the image, the top left coordinate, bottom right coordinate, 
            #color, and line thickness.
            cv2.circle(image_og,(cX,cY), 10, (0,255,255), -1) #BGR
            #The parameters here are the image/frame, the center of the circle, the radius, color, and then thickness. Notice we have a -1 for thickness. 
            #This means the object will actually be filled in, so we will get a filled in circle.
        #else:
            #cv2.rectangle(image_og,(cX-32,cY-32),(cX+32,cY+32),(0,255,0),3)
            #cv2.circle(image_og,(cX,cY), 5, (0,255,0), -1)
            #continue
        
#        savepath = os.path.join(img_save_dir, ('%s.png' %(fnameroot)))
#        #print(savepath)
#        cv2.imwrite(savepath, image_og)
#        
#        if not cv2.imwrite(savepath, image_og):
#            raise Exception("Could not write image")
        
if __name__ == '__main__':

    args=get_args()
    
    og_filenames=[f for f in sorted(os.listdir(args.og_dir)) if not f.endswith('_labels.png')]
    eval_filenames=[f for f in sorted(os.listdir(args.eval_dir)) if f.endswith('.png')]
    
    #og_filenames=og_filenames[0]
    with open('C:/CRC project/CRC_pytorch_chenyu/src/coarseclassifier_40x_64_jitter_e80_more_f3_pred_tumorcell_test_nuclei.txt', 'r') as f:
        a = f.readlines()
    count = 0
    for i in range(len(og_filenames)):
        image_og = cv2.imread(os.path.join(args.og_dir,og_filenames[i]))
        fnameroot=og_filenames[i].replace('.png','')
        img_sg_filenames=[f for f in sorted(eval_filenames) if f.startswith(fnameroot)]
        classlist = a[count:count+len(img_sg_filenames)]
        count = count + len(img_sg_filenames)
        #print(img_sg_filenames)
        print(count)
        
        mark(img_sg_filenames, image_og, fnameroot, classlist)
        
