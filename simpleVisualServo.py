# Make sure to have the add-on "ZMQ remote API" running in
# CoppeliaSim and have following scene loaded:
#
# Franka.ttt
#
# Do not launch simulation, but run this script
#
# All CoppeliaSim commands will run in blocking mode (block
# until a reply from CoppeliaSim is received). For a non-
# blocking example, see simpleTest-nonBlocking.py

import time
import numpy as np
import cv2
import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

# Get Franka's object and script handle
executedMovId = 'notReady'
targetArm = '/Franka'
stringSignalName = targetArm + '_executedMovId'
targetDummy = targetArm + '/target'
objHandle = sim.getObject(targetArm)
scriptHandle = sim.getScript(sim.scripttype_childscript, objHandle)
targetHandle = sim.getObject(targetDummy)

# Get vision sensor's handle
visionSensorHandle = sim.getObject('/Franka/Vision_sensor')
# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Total simulation time
simTime = 60
# Simulation time step
simStep = 0.05
# Synchronous mode
client.setStepping(True)

# Start simulation
sim.startSimulation()
client.step()

# Get vision sensor's parameters
resX = sim.getObjectInt32Param(visionSensorHandle, sim.visionintparam_resolution_x)
resY = sim.getObjectInt32Param(visionSensorHandle, sim.visionintparam_resolution_y)
fov = sim.getObjectFloatParam(visionSensorHandle, sim.visionfloatparam_perspective_angle)
d = 1.3245 - 0.75  # directly measured from scene, should use depth camera measurement instead
kx = ky = 2*d*math.tan(fov/2)/resX  # scaling factor from pixel to meter at d

# Counter
cnt = 0
error_sum = np.zeros(2)
while (t := sim.getSimulationTime()) < simTime:
    img, resX, rexY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
    # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
    # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    # Extract green parts
    green_lower = np.array([50, 100, 100])
    green_upper = np.array([70, 255, 255])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(img_hsv, green_lower, green_upper)

    # Green-masked gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_and(img_gray, img_gray, mask=mask_green)
    img_gray = cv2.medianBlur(img_gray, 5)

    # Detect circle
    rows = img_gray.shape[0]
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=40, param2=15, minRadius=1, maxRadius=100)
    if circles is not None:
        # circles = np.uint16(np.around(circles))
        if cnt > 0:
            old_center = center
        center = np.array([circles[0, 0, 0], circles[0, 0, 1]])
        radius = circles[0, 0, 2]

        # Draw detected circle
        cv2.circle(img, center.astype(int), int(radius), (255, 0, 255), 3)

        # Draw arrow from frame center to circle center
        cv2.arrowedLine(img, (128, 128), center.astype(int), (0, 0, 255), 3)

        # Calculate pixel error in world frame
        circle_camera = np.array(center)
        T_camera_to_world = np.array([[0, -1], [-1, 0]])
        circle_world = np.matmul(T_camera_to_world, circle_camera)
        hand_world = np.array([-resY / 2, -resX / 2])
        if cnt > 0:
            old_error_world = error_world
            error_world = circle_world - hand_world
        else:
            old_error_world = circle_world - hand_world
            error_world = old_error_world
        error_sum += error_world

        # Check convergence
        tol = 1
        if abs(error_world[0]) <= tol:
            error_world[0] = 0
            old_error_world[0] = 0
            flagX = True
        else:
            flagX = False
        if abs(error_world[1]) <= tol:
            error_world[1] = 0
            old_error_world[1] = 0
            flagY = True
        else:
            flagY = False

        # Optical flow of circle center
        dP = 0
        if cnt > 0:
            flow_camera = center - old_center
            flow_world = kx*np.matmul(T_camera_to_world, flow_camera)
            # dP += flow_world

        # Get current pose
        currentPose = sim.getObjectPose(targetHandle, sim.handle_world)

        # Calculate dP
        kP = 0.01
        kI = 0
        kD = 0
        dP += (kP * error_world + kI * error_sum + kD * (error_world - old_error_world)) * simStep

        targetP = np.array(currentPose[0:2]) + dP

        if not flagX or not flagY:
            targetPose = np.concatenate((targetP, np.array([1.3245, 0, 0, 0, 1]))).tolist()
        else:
            targetPose = currentPose

        sim.setObjectPose(targetHandle, sim.handle_world, targetPose)

    cv2.imshow('', img)
    cv2.waitKey(1)

    client.step()
    cnt = cnt + 1

sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

cv2.destroyAllWindows()

print('Program ended')
