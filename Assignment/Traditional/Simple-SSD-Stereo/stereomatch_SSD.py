#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------

# Copyright (c) 2016 David Christian
# Licensed under the MIT License
import numpy as np
from PIL import Image
import os
from KITTIloader2015 import dataloader
from KITTIloader2015 import compute_depth_metrics_i
import matplotlib.pyplot as plt
import cv2

def stereo_match(left_img, right_img, kernel, max_offset, iter, result):
    # Load in both images, assumed to be RGBA 8bit per channel images
    # left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    # right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    print(left.shape)
    w, h = left.shape[1], left.shape[0]  # assume that both images are same size
    
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half = int(kernel / 2)    
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
      
    for y in range(kernel_half, h - kernel_half):      
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            
            for offset in range(max_offset):               
                ssd = 0
                ssd_temp = 0                            
                
                # v and u are the x,y of our local window search, used to ensure a good 
                # match- going by the squared differences of two pixels alone is insufficient, 
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster 
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                        ssd += ssd_temp * ssd_temp              
                
                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
                            
            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_metrics_i(depth, result)
    print(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    # Convert to PIL and save it
    name = 'depth' + str(iter) + '.png'
    Image.fromarray(depth).save(name)

# def load_data(path):
#     data = []
#     for image in os.listdir(path):
#         # if image.find('cat') != -1:
#         #     label = '0'  # cat
#         # elif image.find('dog') != -1:
#         #     label = '1'  # dog
#         # else:
#         #     raise Exception('One of the files is not named cat/dog', image)
#         f_path = os.path.join(path, image)
#         image = Image.open(f_path)
#         image = image.convert('L')
#         # image = image.resize((img_size, img_size), Image.ANTIALIAS)
#         data.append([np.array(image)])
#     return np.array(data)

if __name__ == '__main__':
    left_train, right_train, disp_train_L = dataloader('./Training/')
    # index = 0
    print(len(left_train))
    for index in range(0,len(left_train)):
        img1 = right_train[index]
        img = left_train[index]
        # # print(img)
        # plt.figure()
        # plt.axis('off')
        # plt.imshow(img, cmap='gray')
        # # plt.scatter(x=coord[:, 0], y=coord[:, 1], marker='x', c='b', s=20)
        # plt.show()
        # plt.figure()
        # plt.axis('off')
        # plt.imshow(disp_train_L[index], cmap='gray')
        # # plt.scatter(x=coord[:, 0], y=coord[:, 1], marker='x', c='b', s=20)
        # plt.show()
        # cv2.imshow('image_left', img)
        # cv2.imshow('image_right', img1)

        # Image.fromarray(disp_train_L[index]).save('Disp_L_'+str(index)+'.png')
        stereo_match(img, img1, 3, 30, index, disp_train_L[index])  # 6x6 local search kernel, 30 pixel search range
        index = index + 1

    # for index in range(0, len(left_train)):
