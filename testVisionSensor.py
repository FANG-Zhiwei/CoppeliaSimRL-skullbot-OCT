'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk/ afjsliny@gmail.com
Date: 2024-03-19 15:53:39
LastEditTime: 2024-04-01 16:58:20
Description: To test the communication && input of the vision sensor
'''
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import cv2
from PIL import Image
import math

def minEdgeDist(Image, resX, resY):
    edgeThreshold = 1
    onEdge = False
    minDist = 1e6
    tempDist = 1e6
    point = [0, 0]
    for row in range(resX):
        for column in range(resY):
            if Image[row, column] > edgeThreshold:
                onEdge = True
                tempDist = np.sqrt((row-resX/2)**2 + (column-resX/2)**2)
                tempDist = np.round(tempDist, 2)
                if tempDist < minDist:
                    minDist = tempDist
                    point = [row, column]
            # else:
            #     onEdge = False
    return onEdge, minDist, point


client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)


sim.startSimulation()

# get the handle of the vision sensor
vision_sensor_handle = sim.getObjectHandle('sim_OCT_vision_sensor')

while True: 

    # get the image from the vision sensor
    # image, reso = sim.getVisionSensorImg(vision_sensor_handle)
    image, resX, resY = sim.getVisionSensorCharImage(vision_sensor_handle)
    print(resX, resY)
    # print(image)

    imageMatrix = np.frombuffer(image, dtype=np.uint8).reshape(resY, resX, 3)
    imageMatrix = cv2.flip(cv2.cvtColor(imageMatrix, cv2.COLOR_RGB2BGR), 0)
    # imageMatrix = cv2.flip(imageMatrix, 0)

    # print(imageMatrix.shape)
    cv2.imshow('image', imageMatrix)
    cv2.waitKey(0)

    gray_image = cv2.cvtColor(imageMatrix, cv2.COLOR_RGB2GRAY)
    
    cv2.imshow('gray', gray_image)
    cv2.waitKey(0)

    '''edge detection'''
    canny = cv2.Canny(gray_image, 100, 175)
    # print(canny)
    # print(canny.shape)
    maxCanny = np.max(canny)
    minCanny = np.min(canny)


    _, minDist, _ = minEdgeDist(canny, resX, resY)
    halfBlankSize = math.ceil(minDist/10)*10

    outerImage = canny.copy()
    outerImage[int(resX//2)-halfBlankSize:int(resX//2)+halfBlankSize, 
            int(resY//2)-halfBlankSize:int(resY//2)+halfBlankSize] = 0
    _, minDist, closestPoint = minEdgeDist(outerImage, resX, resY)
    print(minDist)




    cannyImg = Image.fromarray(canny)
    cannyImg.show()
            
    cv2.waitKey(0)
    sim.step()
sim.stopSimulation()
    

