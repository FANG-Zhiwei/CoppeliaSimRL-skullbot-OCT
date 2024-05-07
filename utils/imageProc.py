'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk/ afjsliny@gmail.com
Date: 2024-03-26 16:34:28
LastEditTime: 2024-04-23 13:56:26
Description: 
'''
import numpy as np
import time
import cv2
from PIL import Image
import math
import matplotlib.pyplot as plt
import io

edgeThreshold = 150
kernel_size = 3
half_kernel_size = kernel_size // 2

def visionSensorImage2Canny(image, resX, resY):
    # startTime = time.time()
    imageMatrix = np.frombuffer(image, dtype=np.uint8).reshape(resY, resX, 3)
    imageMatrix = cv2.flip(cv2.cvtColor(imageMatrix, cv2.COLOR_RGB2BGR), 0)

    gray_image = cv2.cvtColor(imageMatrix, cv2.COLOR_RGB2GRAY)
    cannyImg = cv2.Canny(gray_image, 100, 175)
    # endTime = time.time()
    # print('Time for Canny edge detection:', endTime-startTime)
    return cannyImg

def minEdgeDist(image, resX, resY):
    # startTime = time.time()
    onEdge = False
    minDist = 1e6
    point = [0, 0]

    edges = np.where(image >= edgeThreshold)
    rows = edges[0]
    columns = edges[1]
    if len(rows) * len(columns) == 0:
        onEdge = False
    else:
        onEdge = True
        row_dist = (rows-resX/2)**2
        column_dist = (columns-resY/2)**2
        dist = row_dist + column_dist
        minDist = np.sqrt(np.min(dist))
        minIndex = np.argmin(dist)
        point = [rows[minIndex], columns[minIndex]]

    # region
    # forstartTime = time.time()
    # for row in (rows):
    #     for column in (columns):
    #         if image[row, column] >= edgeThreshold:
    #             onEdge = True
    #             tempDist = np.sqrt(row_dist[row] + column_dist[column])
    #             tempDist = np.round(tempDist, 2)
    #             if tempDist < minDist:
    #                 minDist = tempDist
    #                 point = [row, column]
    

            # else:
            #     onEdge = False
    # print('point value', image[point[0], point[1]], point)
    # forendTime = time.time()
    # print('for Time in minEdgeDist:', forendTime-forstartTime)
    # endregion

    # endTime = time.time()
    # print('Time for finding minEdgeDist:', endTime-startTime)
    return onEdge, minDist, point, edges

def secondEdgeDist(image, resX, resY, minDist):
    # startTime = time.time()

    halfBlankSize = math.ceil(minDist/10)*12
    outerImage = image.copy()
    outerImage[int(resX//2)-halfBlankSize:int(resX//2)+halfBlankSize, 
            int(resY//2)-halfBlankSize:int(resY//2)+halfBlankSize] = 0
    _, edgeDist, point, edges = minEdgeDist(outerImage, resX, resY)
    # endTime = time.time()
    # print('Time for finding secondEdgeDist:', endTime-startTime)
    return outerImage, edgeDist, point, edges

def calculateTangetVector(edges, point):
    # startTime = time.time()
    p2 = [0, 0] # the second point on the curve (image)
    dist = (p2[0] - point[0])**2 + (p2[1] - point[1])**2 # initial distance between two points
    
    rows = edges[0]
    columns = edges[1]
    if len(rows) * len(columns) == 0:
        onEdge = False
    else:
        onEdge = True
        row_dist = (rows-point[0])**2
        column_dist = (columns-point[1])**2
        dist = row_dist + column_dist
        mask=dist > 0
        dist = dist[mask]
        minIndex = np.argmin(dist)
        p2 = [rows[minIndex], columns[minIndex]]

    # region
    # rows = np.arange(resX)
    # columns = np.arange(resY)
    # row_dist = (rows-point[0])**2
    # column_dist = (columns-point[1])**2
    # forTIme = time.time()
    # for row in rows:
    #     for column in columns:
    #         if image[row, column] >= edgeThreshold:
    #             if  row_dist[row] + column_dist[column] <= dist:
    #                 p2 = [row, column]
    # endforTime = time.time()
    # print('for Time in calculateTangent:', endforTime-forTIme)
    # endregion
    # print(point, p2)
    tangent_vector = [p2[0] - point[0], p2[1] - point[1]]
    length = np.linalg.norm(tangent_vector)
    tangent_vector = (tangent_vector /length) if length!=0 else [0.0, 0.0]
    tangent_vector = np.array(tangent_vector)
    # endTime = time.time()
    # print('Time for calculating tangent vector:', endTime-startTime)
    return tangent_vector

if __name__ == "__main__":

    # resX=256
    # resY=256
    # image = np.zeros((resX, resY, 3), dtype=np.uint8)
    # image[:, :, :] = np.random.randint(0, 256, (resX, resY, 3))
    image = Image.open('./pics/3.png')
    resX, resY = image.size
    image = np.array(image, dtype=np.uint8)


    startTime=time.time()
    cannyImg = visionSensorImage2Canny(image, resX, resY)
    _, minDist, point1, _ = minEdgeDist(cannyImg, resX, resY)
    print('minDist:', minDist, 'point1:', point1   )
    outerImage, edgeDist, nearest_point, edges = secondEdgeDist(cannyImg, resX, resY, minDist)
    holdingDist = 2.5*minDist
    tangentVector = calculateTangetVector(edges, nearest_point)*60
    print('tangentVector:', tangentVector)
    # print(tangentVector[0], tangentVector[1])
    # # draw the tangent vector
    origin= nearest_point
    end_point = (origin + tangentVector).astype(int)
    translation_matrix = np.float32([[1, 0, tangentVector[1]], [0, 1, tangentVector[0]]])
    target_image = cv2.warpAffine(outerImage, translation_matrix, (outerImage.shape[1], outerImage.shape[0]))
    imageLoss_1 = (np.sum(np.abs(outerImage - target_image))) / 256
    image1Time = time.time()
    cannyImg=cv2.circle(cannyImg, tuple([nearest_point[1],nearest_point[0]]), 20, (255), 3)
    cannyImg=cv2.circle(cannyImg, tuple([point1[1], point1[0]]), 10, (255), -1)
    cannyImg=cv2.arrowedLine(cannyImg, tuple([origin[1], origin[0]]), tuple([end_point[1], end_point[0]]), (255), 6)
    cv2.imwrite('cannyImage.jpg', cannyImg)
    cv2.imwrite('outerImage.jpg', outerImage)
    cv2.imwrite('target_image.jpg', target_image)

   
    image_after = np.zeros((resX, resY, 3), dtype=np.uint8)
    image_after[:, :, :] = np.random.randint(0, 256, (resX, resY, 3))
    cannyImg_after = visionSensorImage2Canny(image_after, resX, resY)
    _, minDist_after, _, _  = minEdgeDist(cannyImg_after, resX, resY)
    outerImage_after, _, _, _ = secondEdgeDist(cannyImg_after, resX, resY, minDist_after)

    imageLoss_2 = (np.sum(np.abs(outerImage_after - target_image))) / 256
    imageLoss_3 = (np.sum(np.abs(outerImage_after - outerImage))) / 256
    image2Time = time.time()

    # print(startTime, image1Time, image2Time)
    print('total time', image2Time - startTime)