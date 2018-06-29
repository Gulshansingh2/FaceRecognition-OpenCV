# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:05:15 2018

@author: GUL
"""
# In[1]:
import numpy as np
import os
import math
from matplotlib import pyplot as plt
import cv2

# In[2]:

webcam = cv2.VideoCapture(0)
ret, frame = webcam.read()
print(ret)
webcam.release

# In[3]

#open a new thread to manage the external cv2 interaction

cv2.startWindowThread()

#Create a window holder to show your image in
cv2.namedWindow("OpenCV window", cv2.WINDOW_NORMAL)
cv2.imshow("opencv frame",frame)

cv2.waitKey()
cv2.destroyAllWindows()

# In[4]
# Open CV --> BGR
# Matlab --> RGB
def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.axis("off") 
        plt.imshow(image, cmap="Greys_r")
        plt.show()

# In[5]
cv2.imwrite('C:\\Users\\GUL\\AnacondaProjects\\FaceRecogOpenCV\\image\\pic_GBR.jpg', frame)
cv2.imwrite('C:\\Users\\GUL\\AnacondaProjects\\FaceRecogOpenCV\\image\\pic_RGB.jpg', frame_RGB)

#os.system("nautilus images")

# In[6]
webcam = cv2.VideoCapture(0)
print (webcam.isOpened())

# In[7]
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)

while True:
    
    _, frame = webcam.read()
    cv2.imshow("image", frame)
    
    #code 271 is ESC key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
    
# In[8]

from IPython.display import clear_output
try:
    while True:
        _, frame = webcam.read()
        plt_show(frame)
        clear_output(wait=True)
except KeyboardInterrupt:
    print("Live Video Interrupted")
    
# In[9]
webcam.release()    

# In[10]

webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
message = ""

while webcam.isOpened():
    
    _, frame = webcam.read()
    
    cv2.rectangle(frame,(100,100), (530,400), (150,150,0), 3 )
    cv2.putText(frame, message, (95, 95), cv2.FONT_HERSHEY_SIMPLEX, .7, 
                (150, 150, 0), 2)
    
    cv2.imshow('PyData Tutorial',frame)
    key = cv2.waitKey(100) & 0xFF
    if key not in [255, 27]:
        message += chr(key)
    elif key == 27:
        break
        
# release both video objects created
webcam.release()
cv2.destroyAllWindows()

# In[11]

webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)

while webcam.isOpened():
    
    _, frame = webcam.read()
    mask = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    cv2.circle(mask, (int(width / 2), int(height / 2)), 200, (255, 255, 255), -1)
    frame = np.bitwise_and(frame, mask)
    
    cv2.imshow('PyData Tutorial', frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
        
# release both video objects created
webcam.release()
cv2.destroyAllWindows()

# In[12]

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global x_in, y_in
    if event == cv2.EVENT_LBUTTONDOWN:
        x_in = x 
        y_in = y
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(frame, (int(int((x + x_in)) / 2), int(int((y + y_in)/2))), 
                   int(math.sqrt((y - y_in) ** 2 + (x - x_in) ** 2) / 2), (150, 150, 0), -1)
        
cv2.namedWindow('PyData Tutorial')
cv2.setMouseCallback('PyData Tutorial', draw_circle)

webcam = cv2.VideoCapture(0)
_, frame = webcam.read()
webcam.release()

while True:
    cv2.imshow('PyData Tutorial',frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
