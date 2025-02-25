#!/usr/bin/env python3

from frame2d import Frame2D
from cmap import CozmoMap, is_in_map, Coord2D
import math

import numpy as np

cozmoOdomNoiseX = 0.002
cozmoOdomNoiseY = 0.002
cozmoOdomNoiseTheta = 0.001

wheeldistance = 70 #mm , determined experimentally

# Forward kinematics: compute coordinate frame update as Frame2D from left/right track speed and time of movement
def track_speed_to_pose_change(left, right, time):
    # TODO
	l1 = left*time
	l2 = right*time
	if left==right:
		x=l1
		y=0
		a=0		
	else: 	
		theta = (l2-l1)/wheeldistance
		r = (l1+l2)/ (2*theta)
		x = r*math.sin(theta)
		y = -r*(math.cos(theta)-1)
		a = theta	
	return Frame2D.fromXYA(x,y,a)

# Differential inverse kinematics: compute left/right track speed from desired angular and forward velocity
def velocity_to_track_speed(forward, angular):
    # TODO
	left = forward - angular*wheeldistance/2
	right = forward + angular*wheeldistance/2
	return [left,right]

def cliff_sensor_model(robotPose : Frame2D, m : CozmoMap, cliffDetected):
	sensorPose = robotPose.mult(Frame2D.fromXYA(20,0,0))
	if not is_in_map(m, robotPose.x(), robotPose.y()):
		return 0
	if not is_in_map(m, sensorPose.x(), sensorPose.y()):
		return 0
	c = Coord2D(sensorPose.x(), sensorPose.y())
	if m.grid.isOccupied(c) == cliffDetected: # TODO this will not always be exact
		return 1.0
	else:
		return 0.0

# Take a true cube position (relative to robot frame). 
# Compute /probability/ of cube being (i) visible AND being detected at a specific measure position (relative to robot frame)
def cube_sensor_model(expectedCubePosition, visible, measuredCubePosition):
	# TODO Implement a sensor model for the cube
	# This should return a value between 0 and 1
	# visible is a boolean indicating if the cube is visible
	# measuredPosition is the position of the cube relative to the robot
	# trueCubePosition is the true position of the cube relative to the robot
	if visible:
		measuredX = measuredCubePosition.x()
		measuredY = measuredCubePosition.y() #consider taking absolute value
		measuredA = measuredCubePosition.angle()

		expectedX = expectedCubePosition.x()
		expectedY = expectedCubePosition.y()
		expectedA = expectedCubePosition.angle()

		sigmaX = 30 #mm
		sigmaY = 30 #mm
		sigmaA = 0.25 #radian
			
		xDeviation = measuredX - expectedX
		yDeviation = measuredY - expectedY
		aDeviation = measuredA - expectedA

		xError = xDeviation * xDeviation / (sigmaX * sigmaX)
		yError = yDeviation * yDeviation / (sigmaY * sigmaY)
		aError = aDeviation * aDeviation / (sigmaA * sigmaA)

		maxX = 700 
		maxY = 700  # Temporary placeholders
		maxA = 2 * math.pi  # Different for walls

		minX = 50
		minY = 50
		minA = - 2 * math.pi  # Different for walls

		#N = 1 / ((sigmaX * sigmaY * sigmaA) * (2 * math.pi) ** (3 / 2))  # Normalization factor
		N = 1
		pVisible = N * np.exp(-0.5 * (xError + yError + aError))
		#print(pVisible)

		#if measuredX > maxX or measuredY > maxY or measuredA > maxA: #need to account for negative values
		#	pVisible = 0  # Zero probability outside measurable range
		#if measuredX < minX or measuredY < minY or measuredA < minA: #need to account for negative values 
		#	pVisible = 0 

		if visible:
			return pVisible
		else:
			return 1.0 - pVisible
	else:
		return 1.0 #cube is not visible and makes other postions plausible???? No cube detected, I could be anywhere! 

'''
Handling Invisible Cubes

    Right now, if a cube is not visible (visible == False), your function returns 0.0.
    The problem is that in a Monte Carlo Localisation (MCL) framework, probabilities are multiplied across multiple sensor observations.
    If even one cube is invisible, returning 0.0 will cause the entire probability product to be zero, which essentially wipes out all hypotheses.
    Instead, when a cube is invisible, the function should return 1.0 so that it does not negatively impact the overall probability.
'''


