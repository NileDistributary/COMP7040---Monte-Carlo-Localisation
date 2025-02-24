#!/usr/bin/env python3

import asyncio
import cozmo
from frame2d import Frame2D 
from map import CozmoMap, plotMap, loadU08520Map, Coord2D
from matplotlib import pyplot as plt
from cozmo_interface import cube_sensor_model, cliff_sensor_model
from cozmo_interface import track_speed_to_pose_change,cozmoOdomNoiseX,cozmoOdomNoiseY,cozmoOdomNoiseTheta
from mcl_tools import *
from gaussian import Gaussian, GaussianTable, plotGaussian
from cozmo.util import degrees, distance_mm, speed_mmps
import math
import numpy as np
import threading
import time
import os

m = loadU08520Map()
mapPrior = Uniform(
		np.array([m.grid.minX(),m.grid.minY(),0]),
		np.array([m.grid.maxX(),m.grid.maxY(),2*math.pi]))

numParticles = 50
particles = sampleFromPrior(mapPrior, numParticles)
xyaResampleVar = np.diag([10,10,10*math.pi/180])
xyaResampleNoise = GaussianTable(np.zeros([3]),xyaResampleVar, 10000)
xyaNoiseVar = np.diag([cozmoOdomNoiseX,cozmoOdomNoiseY,cozmoOdomNoiseTheta])
xyaNoise = GaussianTable(np.zeros([3]),xyaNoiseVar,10000)

def runMCLLoop(robot: cozmo.robot.Robot):
	global particles
	
	particleWeights = np.zeros([numParticles])
	cubeIDs = [cozmo.objects.LightCube1Id,cozmo.objects.LightCube2Id,cozmo.objects.LightCube3Id]

	timeInterval = 0.1
	t = 0
	while True:
		t0 = time.time()
		
		robotPose = Frame2D.fromPose(robot.pose)
		cubeVisibility = {}
		cubeRelativeFrames = {}
		numVisibleCubes = 0
		for cubeID in cubeIDs:
			cube = robot.world.get_light_cube(cubeID)
			relativePose = Frame2D()
			visible = False
			if cube is not None and cube.is_visible:
				cubePose = Frame2D.fromPose(cube.pose)
				relativePose = robotPose.inverse().mult(cubePose)
				visible = True
				numVisibleCubes = numVisibleCubes+1
			cubeVisibility[cubeID] =  visible
			cubeRelativeFrames[cubeID] =  relativePose

		cliffDetected = robot.is_cliff_detected
		lspeed = robot.left_wheel_speed.speed_mmps
		rspeed = robot.right_wheel_speed.speed_mmps
		currentParticles = particles

		poseChange = track_speed_to_pose_change(lspeed,rspeed,timeInterval)
		poseChangeXYA = poseChange.toXYA()
		for i in range(0,numParticles):
			poseChangeXYAnoise = np.add(poseChangeXYA, xyaNoise.sample())
			poseChangeNoise = Frame2D.fromXYA(poseChangeXYAnoise)
			currentParticles[i] = currentParticles[i].mult(poseChangeNoise)

		for i in range(0,numParticles):
			p = 1.0
			for cubeID in cubeIDs:
				relativeTruePose = currentParticles[i].inverse().mult(m.landmarks[cubeID])
				p = p * cube_sensor_model(relativeTruePose, cubeVisibility[cubeID], cubeRelativeFrames[cubeID])
			p = p * cliff_sensor_model(currentParticles[i], m, cliffDetected)
			particleWeights[i] = p

		freshParticlePortion = 0.1
		numFreshParticles = int(numParticles*freshParticlePortion)
		numResampledParticles = numParticles - numFreshParticles
		#resampledParticles = resampleIndependent(currentParticles, particleWeights, numResampledParticles, xyaResampleNoise)
		resampledParticles = resampleLowVar(currentParticles, particleWeights, numResampledParticles, xyaResampleNoise)
		freshParticles = sampleFromPrior(mapPrior, numFreshParticles)
		newParticles = resampledParticles + freshParticles
		particles = newParticles

		print("t = "+str(t))
		t = t+1

		t1 = time.time()
		timeTaken = t1-t0
		if timeTaken < timeInterval:
			time.sleep(timeInterval - timeTaken)
		else:
			print("Warning: loop iteration tool more than "+str(timeInterval) + " seconds (t="+str(timeTaken)+")")

def runPlotLoop(robot: cozmo.robot.Robot, method_name="Independent", save_interval=0.5):
    global particles

    # Setup for interactive plotting
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    ax.set_xlim(m.grid.minX(), m.grid.maxX())
    ax.set_ylim(m.grid.minY(), m.grid.maxY())

    plotMap(ax, m)

    particlesXYA = np.zeros([numParticles, 3])
    for i in range(0, numParticles):
        particlesXYA[i, :] = particles[i].toXYA()

    particlePlot = plt.scatter(particlesXYA[:, 0], particlesXYA[:, 1], color="red", zorder=3, s=10, alpha=0.5)

    empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
    gaussianPlot = plotGaussian(empiricalG, color="red")

    # Create a directory for saving the plots, if not exist
    output_dir = f"plots/{method_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Track time (in seconds)
    start_time = time.time()
    t = 0

    # Store references for text objects
    method_text = None
    timestamp_text = None

    while True:
        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Save the plot at specified intervals
        if elapsed_time >= t * save_interval:
            # Update the plot
            for i in range(0, numParticles):
                particlesXYA[i, :] = particles[i].toXYA()
            particlePlot.set_offsets(particlesXYA[:, 0:2])

            empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
            plotGaussian(empiricalG, color="red", existingPlot=gaussianPlot)

            # Remove previous text labels
            if method_text is not None:
                method_text.remove()
            if timestamp_text is not None:
                timestamp_text.remove()

            # Add the method label
            method_text = plt.text(0.05, 0.95, f"Method: {method_name}", transform=ax.transAxes, fontsize=12, verticalalignment='top', color='blue')

            # Add the timestamp to the figure
            interval_timestamp = round(elapsed_time, 1)  # Round to nearest 0.1s
            timestamp_text = plt.text(0.05, 0.9, f"t = {interval_timestamp}s", transform=ax.transAxes, fontsize=12, verticalalignment='top', color='green')

            # Timestamp for the image (based on elapsed time)
            plot_filename = os.path.join(output_dir, f"plot_t{interval_timestamp}s.png")

            # Save the plot
            plt.savefig(plot_filename)

            print(f"Saved plot at t={interval_timestamp}s")

            # Increment the interval counter
            t += 1

        # Pause to prevent the loop from running too fast
        plt.draw()
        plt.pause(0.001)
        time.sleep(0.01)



def cozmo_program(robot: cozmo.robot.Robot):
	robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
	threading.Thread(target=runMCLLoop, args=(robot,)).start()
	threading.Thread(target=lambda: runPlotLoop(robot, method_name="Low Variance", save_interval=0.5)).start()

	robot.enable_stop_on_cliff(True)

	t = 0
	while True:
		print("Straight")
		robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()
		print("Turn")
		robot.turn_in_place(degrees(90),speed=degrees(20)).wait_for_completed()
		print("Sleep")
		time.sleep(1)

cozmo.robot.Robot.drive_off_charger_on_connect = False
cozmo.run_program(cozmo_program, use_3d_viewer=False, use_viewer=False)
