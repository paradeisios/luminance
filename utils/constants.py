#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:17:34 2021

@author: paradeisios
"""
import numpy as np

METHODS = { "linear" :    lambda x: 0.2126*x[:,:,0]+0.7152*x[:,:,1]+0.0722*x[:,:,2],
            "perceived" : lambda x: 0.299*x[:,:,0]+0.587*x[:,:,1]+0.114*x[:,:,2],
            "average" :   lambda x: np.mean(x,2) }