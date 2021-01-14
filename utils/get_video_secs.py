#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:17:00 2021

@author: paradeisios
"""
import cv2

def get_video_secs(video):
    vidcap = cv2.VideoCapture(video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    vidcap.release() 
    return int(float(totalNoFrames) / float(fps))
 