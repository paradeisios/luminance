#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:44:38 2021

@author: paradeisios
"""
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

from utils.frame_luminance import frame_luminance
from utils.get_video_secs import get_video_secs
from utils.constants import METHODS


parser = ArgumentParser(description='Extract global and local luminance')
arguements = parser.add_argument_group('Arguments')

arguements.add_argument('--video_path', metavar='path', nargs=1, type=str,
                      help='Path with video.',
                      required=True)
arguements.add_argument('--pupil_path', metavar='path', nargs=1, type=str,
                      help='Path to pupil data.',
                      required=True)
arguements.add_argument('--out_path', metavar='path', nargs=1, type=str,
                      help='Path to store the results.',
                      required=True)
arguements.add_argument('--method', metavar='method', nargs=1, type=str,
                      default = "linear", required=False,
                      help='Mathematical model to calculate luminance.')
arguements.add_argument('--downsample', metavar='bool', nargs=1, 
			 type=bool, default = True, required=False,
                        help='Return the downsampled luminance.')
 
args = parser.parse_args()
video           = args.video_path
pupil_data_path = args.pupil_path
output_path     = args.out_path
method          = args.method
downsample      = args.downsample

if isinstance(video, list):
    video = video[0]

if isinstance(pupil_data_path, list):
    pupil_data_path = pupil_data_path[0]

if isinstance(output_path, list):
    output_path = output_path[0]

if isinstance(method, list):
    method = method[0]

if isinstance(downsample, list):
    downsample = downsample[0]
    

pupil_data = np.genfromtxt(pupil_data_path,delimiter=",")
pupil_x = pupil_data[:,0].astype(np.int16)
pupil_y = pupil_data[:,1].astype(np.int16)
radius = 100

height = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_WIDTH))
frame_count = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))



luminance_array = np.zeros((frame_count,2))

pbar = tqdm(total=frame_count)
pbar.set_description("Calculating global luminace")
vidcap = cv2.VideoCapture(video)
for ii in range(frame_count):
    _,image = vidcap.read()
    luminance_array[ii,0] = frame_luminance(image,METHODS["linear"])
    pbar.update(1)
pbar.close()
vidcap.release() 


pbar = tqdm(total=frame_count)
pbar.set_description("Calculating local luminace")
vidcap = cv2.VideoCapture(video)
for ii in range(frame_count):
    _,image = vidcap.read()
    mask = np.zeros((height,width), np.uint8)
    mask = cv2.circle(mask,(pupil_x[ii],pupil_y[ii]),radius,1,thickness=-1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    image = np.array(Image.fromarray(image.astype(np.uint8)).convert('RGB')).astype("float64")
    image[image==0]=np.nan
    luminance_array[ii,1] = frame_luminance(masked_image,METHODS["linear"])
    pbar.update(1)
pbar.close()
vidcap.release() 

if downsample:
    
    seconds = get_video_secs(video)
    downsampled = [np.array_split(luminance_array[:,ii], seconds) \
               for ii in range(luminance_array.shape[1])]
    luminance_array = np.array([list(map(np.nanmean,downsampled[ii])) \
               for ii in range(len(downsampled))]).T
    
        
df = pd.DataFrame(luminance_array,columns=["global","local"])
save_name = output_path + "luminance.csv"
df.to_csv(save_name)


