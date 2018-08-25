import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('test2.png')
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(HSV[y,x])
#th2=cv2.adaptiveThreshold(imagegray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("imageHSV",HSV)
cv2.imshow('image',image)
cv2.setMouseCallback("imageHSV",getpos)
cv2.waitKey(0)