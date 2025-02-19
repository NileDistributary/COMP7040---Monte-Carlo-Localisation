#!/usr/bin/env python3

from frame2d import Frame2D
from cmap import CozmoMap, is_in_map, Coord2D
import math

import numpy as np
cozmoOdomNoiseX = 0.1
cozmoOdomNoiseY = 0.1
cozmoOdomNoiseTheta = 0.1

wheeldistance = 80

# Forward kinematics: compute coordinate frame update as Frame2D from left/right track speed and time of movement
def track_speed_to_pose_change(left, right, time):
    # TODO
	if left==right:
		x=l1
		y=0
		a=0		
	else: 	
		l1 = left*time
		l2 = right*time
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
def cube_sensor_model(trueCubePosition, visible, measuredPosition):
    return 1.0


# Take a true wall position (relative to robot frame). 
# Compute /probability/ of wall being (i) visible AND being detected at a specific measure position (relative to robot frame)
def wall_sensor_model(trueWallPosition, visible, measuredPosition):
    return 1.0

''' updated sensor model adds walls to list of sensed objects. We assume here that walls will
    be passed in much like cubes, namely as a dict of visible ones and a dict of (left,centre,right) frame
    triplets. We expect the visibility list AND the frame list to be in the Cozmo ID system rather than the
    simulation ID  system (i.e. walls not yet even detected aren't here). There will have to be a
    transformation at the user side from the returned wall poses in visible_walls() to the relative frames
    passed in here (or we could set this up so that visible_walls returns Frames rather than poses)'''
def cozmo_sensor_model(robotPose : Frame2D, m : CozmoMap, cliffDetected, cubeVisibility, cubeRelativeFrames, wallVisibility, wallRelativeFrames):
        p = 1.
        #for cubeID in cubeVisibility:
                #cube_sensor_model(.....) # FIXME use result correctly
        #for wallCID in wallVisibility:
                #wall_sensor_model(.....) # FIXME use result correctly
        p = p * cliff_sensor_model(robotPose, m, cliffDetected)
        return p

