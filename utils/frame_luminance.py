#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:15:25 2021

@author: paradeisios
"""
import numpy as np
import warnings
from PIL import Image

def frame_luminance(image,method):
   
     with warnings.catch_warnings():
         warnings.filterwarnings('error')
         try:
             image = np.array(Image.fromarray(image.astype(np.uint8)).convert('RGB')).astype("float64")
             image[image==0]=np.nan
             mean =  np.nanmean(method(image))
         except RuntimeWarning:
             mean = np.nan           
     return mean