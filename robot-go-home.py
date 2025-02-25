#!/usr/bin/env python3

import asyncio
import cozmo
import time
import math
import numpy as np
import threading
from matplotlib import pyplot as plt

from frame2d import Frame2D 
from cmap import CozmoMap, plotMap, loadU08520Map, Coord2D, is_in_map
from cozmo_interface import cube_sensor_model, cliff_sensor_model, track_speed_to_pose_change
from cozmo_interface import cozmoOdomNoiseX, cozmoOdomNoiseY, cozmoOdomNoiseTheta
from mcl_tools import *
from gaussian import Gaussian, GaussianTable, plotGaussian
from cozmo.util import degrees, distance_mm, speed_mmps

# Map setup
m = loadU08520Map()
mapPrior = Uniform(
    np.array([m.grid.minX(), m.grid.minY(), 0]),
    np.array([m.grid.maxX(), m.grid.maxY(), 2*math.pi]))

# MCL parameters
numParticles = 200  # Increased for better localization
particles = sampleFromPrior(mapPrior, numParticles)
particleWeights = np.ones([numParticles]) / numParticles  # Initialize with uniform weights

# Noise models
xyaResampleVar = np.diag([5, 5, 5*math.pi/180])  # Reduced from 10 for more stability
xyaResampleNoise = GaussianTable(np.zeros([3]), xyaResampleVar, 10000)
xyaNoiseVar = np.diag([cozmoOdomNoiseX, cozmoOdomNoiseY, cozmoOdomNoiseTheta])
xyaNoise = GaussianTable(np.zeros([3]), xyaNoiseVar, 10000)

# Navigation control flags
localized = False
path_planning_done = False
navigation_complete = False
robot_at_goal = False

# Localization quality metrics
min_effective_particles_ratio = 0.4  # Minimum ratio of effective particles required
max_position_variance = 2500  # Maximum allowable position variance (50^2 mm^2)
max_orientation_variance = 0.1  # Maximum allowable orientation variance (radians^2)
min_visible_cubes = 1  # Minimum number of cubes that must be visible for reliable localization

# Store estimated robot pose
estimated_pose = Frame2D()
position_variance = float('inf')
orientation_variance = float('inf')
effective_particle_ratio = 0

def compute_neff(weights):
    """Calculate the effective number of particles."""
    total = np.sum(weights)
    if total == 0:
        return 0  # Prevent division by zero
    
    normalized_weights = weights / total
    return 1.0 / np.sum(normalized_weights ** 2)

def estimate_robot_pose():
    """Estimate the robot's pose from the particle distribution."""
    global estimated_pose, position_variance, orientation_variance
    
    # Convert particles to numpy array for calculations
    particles_xya = np.zeros([numParticles, 3])
    for i in range(numParticles):
        particles_xya[i, :] = particles[i].toXYA()
    
    # Calculate weighted mean
    if np.sum(particleWeights) > 0:
        weights_normalized = particleWeights / np.sum(particleWeights)
        
        # Handle angular values correctly (circular mean)
        cos_sum = np.sum(weights_normalized * np.cos(particles_xya[:, 2]))
        sin_sum = np.sum(weights_normalized * np.sin(particles_xya[:, 2]))
        mean_angle = np.arctan2(sin_sum, cos_sum)
        
        # Mean position (weighted)
        mean_x = np.sum(weights_normalized * particles_xya[:, 0])
        mean_y = np.sum(weights_normalized * particles_xya[:, 1])
        
        # Update estimated pose
        estimated_pose = Frame2D.fromXYA(mean_x, mean_y, mean_angle)
        
        # Calculate variances
        position_variance = np.sum(weights_normalized * ((particles_xya[:, 0] - mean_x)**2 + 
                                                        (particles_xya[:, 1] - mean_y)**2))
        
        # Angular variance calculation needs special handling due to circular nature
        angle_diffs = particles_xya[:, 2] - mean_angle
        # Normalize angle differences to [-pi, pi]
        angle_diffs = np.mod(angle_diffs + np.pi, 2*np.pi) - np.pi
        orientation_variance = np.sum(weights_normalized * angle_diffs**2)
    
    return estimated_pose

def is_well_localized():
    """Determine if the robot is sufficiently localized."""
    global localized, effective_particle_ratio
    
    # Calculate the effective number of particles
    neff = compute_neff(particleWeights)
    effective_particle_ratio = neff / numParticles
    
    # Get the current estimated pose and variances
    current_pose = estimate_robot_pose()
    
    print(f"Localization metrics - Position variance: {position_variance:.2f}, "
          f"Orientation variance: {orientation_variance:.4f}, "
          f"Effective particle ratio: {effective_particle_ratio:.2f}")
    
    # Check if localization criteria are met
    if (effective_particle_ratio >= min_effective_particles_ratio and 
            position_variance <= max_position_variance and 
            orientation_variance <= max_orientation_variance):
        if not localized:
            print("Robot is now well localized!")
        localized = True
        return True
    else:
        if localized:
            print("Lost localization. Restarting localization process...")
        localized = False
        return False

def is_point_in_obstacle(x, y):
    """Check if a point is in an obstacle."""
    if not is_in_map(m, x, y):
        return True  # Points outside the map are considered obstacles
    
    # Check if the point is in an occupied cell
    return m.grid.isOccupied(Coord2D(x, y))

def is_path_clear(start_x, start_y, end_x, end_y, check_steps=10):
    """Check if there's a clear path between two points."""
    for i in range(check_steps + 1):
        t = i / check_steps
        x = start_x + t * (end_x - start_x)
        y = start_y + t * (end_y - start_y)
        
        if is_point_in_obstacle(x, y):
            return False
    return True

def plan_path_to_goal():
    """Plan a path from current position to the goal."""
    global path_planning_done
    
    if not localized:
        return None
    
    # Get current estimated position
    current_x = estimated_pose.x()
    current_y = estimated_pose.y()
    
    # Get goal position (first target in the map)
    goal_x = m.targets[0].x()
    goal_y = m.targets[0].y()
    
    # Simple direct path if clear
    if is_path_clear(current_x, current_y, goal_x, goal_y):
        path_planning_done = True
        return [(goal_x, goal_y)]
    
    # If direct path isn't clear, use a simple waypoint approach to avoid the middle trench
    # The trench is around x=16-32, y=21-23 in grid coordinates (according to map init in loadU08520Map)
    
    # Determine which side of the trench we're on
    if current_y < 350:  # If we're below the trench
        waypoint_x = 250  # Go to a point left of the trench
        waypoint_y = 150  # and below it
        
        # Then go around to the target
        return [(waypoint_x, waypoint_y), (goal_x, goal_y)]
    else:  # If we're above the trench
        waypoint_x = 250  # Go to a point left of the trench
        waypoint_y = 550  # and above it
        
        # Then go to the target
        return [(waypoint_x, waypoint_y), (goal_x, goal_y)]

def navigate_to_waypoint(robot, waypoint, max_speed=50):
    """Navigate to a specific waypoint."""
    current_pose = estimate_robot_pose()
    
    # Extract current position and waypoint coordinates
    current_x = current_pose.x()
    current_y = current_pose.y()
    current_angle = current_pose.angle()
    
    waypoint_x, waypoint_y = waypoint
    
    # Calculate distance and direction to waypoint
    dx = waypoint_x - current_x
    dy = waypoint_y - current_y
    distance = math.sqrt(dx*dx + dy*dy)
    
    # If we're close enough to the waypoint, return success
    if distance < 50:  # 50mm tolerance
        return True
    
    # Calculate target angle to face the waypoint
    target_angle = math.atan2(dy, dx)
    
    # Normalize angle difference to [-pi, pi]
    angle_diff = target_angle - current_angle
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
    
    # First, turn to face the waypoint if needed
    if abs(angle_diff) > 0.1:  # Turn if angle difference is more than ~5.7 degrees
        turn_angle = angle_diff * 180 / math.pi  # Convert to degrees for Cozmo API
        
        print(f"Turning {turn_angle:.1f} degrees to face waypoint")
        robot.turn_in_place(degrees(turn_angle), speed=degrees(30)).wait_for_completed()
        return False  # Return False to indicate we need to continue navigation
    
    # Then drive forward
    # Limit the distance to drive at once
    drive_distance = min(distance, 100)  # Drive at most 100mm at a time
    
    print(f"Driving {drive_distance:.1f}mm toward waypoint")
    robot.drive_straight(distance_mm(drive_distance), speed_mmps(max_speed)).wait_for_completed()
    return False  # Return False to indicate we need to continue navigation

def check_if_at_goal():
    """Check if the robot has reached the goal."""
    global robot_at_goal
    
    current_pose = estimate_robot_pose()
    goal_x = m.targets[0].x()
    goal_y = m.targets[0].y()
    
    # Calculate distance to goal
    dx = goal_x - current_pose.x()
    dy = goal_y - current_pose.y()
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance < 50:  # 50mm tolerance
        robot_at_goal = True
        return True
    return False

def runMCLLoop(robot: cozmo.robot.Robot):
    global particles, particleWeights
    
    cubeIDs = [cozmo.objects.LightCube1Id, cozmo.objects.LightCube2Id, cozmo.objects.LightCube3Id]
    
    # Main MCL loop
    timeInterval = 0.1
    t = 0
    while True:
        t0 = time.time()
        
        # Read cube sensors
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
                numVisibleCubes += 1
            cubeVisibility[cubeID] = visible
            cubeRelativeFrames[cubeID] = relativePose
        
        # Read cliff sensor
        cliffDetected = robot.is_cliff_detected
        
        # Read track speeds
        lspeed = robot.left_wheel_speed.speed_mmps
        rspeed = robot.right_wheel_speed.speed_mmps
        
        # Read global variable
        currentParticles = particles
        
        # MCL step 1: prediction (shift particles through motion model)
        poseChange = track_speed_to_pose_change(lspeed, rspeed, timeInterval)
        poseChangeXYA = poseChange.toXYA()
        for i in range(numParticles):
            poseChangeXYAnoise = np.add(poseChangeXYA, xyaNoise.sample())
            poseChangeNoise = Frame2D.fromXYA(poseChangeXYAnoise)
            currentParticles[i] = currentParticles[i].mult(poseChangeNoise)
        
        # MCL step 2: weighting (weigh particles with sensor model)
        for i in range(numParticles):
            p = 1.0
            for cubeID in cubeIDs:
                relativeTruePose = currentParticles[i].inverse().mult(m.landmarks[cubeID])
                p *= cube_sensor_model(relativeTruePose, cubeVisibility[cubeID], cubeRelativeFrames[cubeID])
            p *= cliff_sensor_model(currentParticles[i], m, cliffDetected)
            particleWeights[i] = p
        
        # MCL step 3: resampling (proportional to weights)
        # Dynamically adjust number of fresh particles based on localization quality
        if np.sum(particleWeights) < 1e-10:
            # If all weights are very low, inject more fresh particles to recover
            freshParticlePortion = 0.5
        else:
            # Check localization quality metrics
            neff = compute_neff(particleWeights)
            effective_ratio = neff / numParticles
            
            if effective_ratio < 0.2:
                # Poor localization - inject more fresh particles
                freshParticlePortion = 0.3
            elif effective_ratio < 0.5:
                # Medium localization quality
                freshParticlePortion = 0.1
            else:
                # Good localization quality - minimal fresh particles
                freshParticlePortion = 0.01
        
        numFreshParticles = int(numParticles * freshParticlePortion)
        numResampledParticles = numParticles - numFreshParticles
        
        # Use low variance resampling for better stability
        resampledParticles = resampleLowVar(currentParticles, particleWeights, numResampledParticles, xyaResampleNoise)
        freshParticles = sampleFromPrior(mapPrior, numFreshParticles)
        newParticles = resampledParticles + freshParticles
        
        # Update global particle set
        particles = newParticles
        
        # Check localization quality
        is_well_localized()
        
        if t % 10 == 0:  # Print every ~1 second
            print(f"t = {t}, Visible cubes: {numVisibleCubes}, Fresh particles: {freshParticlePortion:.2f}")
        
        t += 1
        
        # Adjust loop timing
        t1 = time.time()
        timeTaken = t1 - t0
        if timeTaken < timeInterval:
            time.sleep(timeInterval - timeTaken)
        else:
            print(f"Warning: loop iteration took more than {timeInterval} seconds (t={timeTaken:.4f})")

def runPlotLoop(robot: cozmo.robot.Robot):
    global particles, estimated_pose
    
    # Create plot
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    
    ax.set_xlim(m.grid.minX(), m.grid.maxX())
    ax.set_ylim(m.grid.minY(), m.grid.maxY())
    
    plotMap(ax, m)
    
    # Particle plot
    particlesXYA = np.zeros([numParticles, 3])
    for i in range(numParticles):
        particlesXYA[i, :] = particles[i].toXYA()
    particlePlot = plt.scatter(particlesXYA[:, 0], particlesXYA[:, 1], color="red", zorder=3, s=10, alpha=0.5)
    
    # Estimated pose plot (arrow)
    quiver = ax.quiver([estimated_pose.x()], [estimated_pose.y()], 
                      [math.cos(estimated_pose.angle())], [math.sin(estimated_pose.angle())],
                      color=['blue'], scale=50, zorder=4)
    
    # Uncertainty ellipse
    empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
    gaussianPlot = plotGaussian(empiricalG, color="red")
    
    # Status text
    status_text = ax.text(0.02, 0.98, "Initializing...", transform=ax.transAxes, 
                         verticalalignment='top', fontsize=12)
    
    # Main loop
    while True:
        # Update particle plot
        for i in range(numParticles):
            particlesXYA[i, :] = particles[i].toXYA()
        particlePlot.set_offsets(particlesXYA[:, 0:2])
        
        # Update estimated pose arrow
        quiver.set_offsets(np.array([[estimated_pose.x(), estimated_pose.y()]]))
        quiver.set_UVC(math.cos(estimated_pose.angle()), math.sin(estimated_pose.angle()))
        
        # Update uncertainty ellipse
        empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
        plotGaussian(empiricalG, color="red", existingPlot=gaussianPlot)
        
        # Update status text
        status = "Localizing..."
        if localized:
            status = "Localized - Planning path"
            if path_planning_done:
                status = "Navigating to goal"
                if robot_at_goal:
                    status = "Goal reached!"
        
        status_info = (f"{status}\n"
                      f"Position Var: {position_variance:.1f}\n"
                      f"Angle Var: {orientation_variance:.3f}\n"
                      f"Eff. Particles: {effective_particle_ratio:.2f}")
        status_text.set_text(status_info)
        
        # Refresh plot
        plt.draw()
        plt.pause(0.001)
        
        time.sleep(0.05)

def cozmo_program(robot: cozmo.robot.Robot):
    global path_planning_done, navigation_complete, robot_at_goal
    
    # Set the head angle for better cube visibility
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    
    # Enable cliff detection
    robot.enable_stop_on_cliff(True)
    
    # Start MCL and visualization in separate threads
    threading.Thread(target=runMCLLoop, args=(robot,)).start()
    threading.Thread(target=runPlotLoop, args=(robot,)).start()
    
    # Wait for initial localization
    print("Waiting for initial localization...")
    time.sleep(3)  # Give MCL time to initialize
    
    current_waypoint_index = 0
    path = None
    
    # Main navigation loop
    while not robot_at_goal:
        # Check if robot is at goal
        if check_if_at_goal():
            print("Goal reached! Mission complete.")
            robot.play_anim_trigger(cozmo.anim.Triggers.MajorWin).wait_for_completed()
            break
        
        # If not yet localized, explore to improve localization
        if not localized:
            print("Not well localized. Exploring to improve localization...")
            
            # Simple exploration strategy: turn to look for cubes
            robot.turn_in_place(degrees(45), speed=degrees(20)).wait_for_completed()
            time.sleep(1)
            continue
        
        # If localized but no path planned yet, plan a path
        if path is None:
            print("Planning path to goal...")
            path = plan_path_to_goal()
            if path:
                print(f"Path planned with {len(path)} waypoints")
                current_waypoint_index = 0
                path_planning_done = True
            else:
                print("Could not plan path, will try again")
                time.sleep(1)
                continue
        
        # Navigate to the current waypoint
        current_waypoint = path[current_waypoint_index]
        print(f"Navigating to waypoint {current_waypoint_index+1}/{len(path)}: {current_waypoint}")
        
        # Try to move toward the waypoint
        waypoint_reached = navigate_to_waypoint(robot, current_waypoint)
        
        # If reached the waypoint, move to next waypoint
        if waypoint_reached:
            print(f"Reached waypoint {current_waypoint_index+1}")
            current_waypoint_index += 1
            
            # If all waypoints are reached, we're done
            if current_waypoint_index >= len(path):
                print("All waypoints reached!")
                navigation_complete = True
                check_if_at_goal()  # Final check if we're at the goal
                
                if robot_at_goal:
                    robot.play_anim_trigger(cozmo.anim.Triggers.MajorWin).wait_for_completed()
                    break
                else:
                    # If not at goal despite finishing the path, replan
                    print("Finished path but not at goal. Replanning...")
                    path = None
                    path_planning_done = False
        
        # Pause to allow MCL to update
        time.sleep(0.5)

if __name__ == "__main__":
    cozmo.robot.Robot.drive_off_charger_on_connect = True
    cozmo.run_program(cozmo_program, use_3d_viewer=False, use_viewer=False)
