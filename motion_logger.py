#!/usr/bin/env python3

# Note: this file is UNTESTED at this point. There could be syntax errors etc. as well as
# potentially behavioural issues - some experimentation may be necessary to get it working
# but the structure is hopefully clear.


import asyncio
import sys
import cozmo

from frame2d import Frame2D 

# experimental data
robotFrames = []

# track speed settings for various movements. Each setting is a sequential series of
# commands, consisting of tuples where the first element is the number of time steps
# to run for, the second element are the [left, right] track speeds. Track speed values
# can (should) be adjusted to get close-to-expected behaviour

# this one just turns in a circle - useful for establishing angular errors and effective wheel diameter 
trackSpeedsCircle = [(600, [10,25])]
# this one starts with a curve, goes straight, then does a reverse curve to finish
trackSpeedsCurvy = [(200, [10, 25]), (200, [10,10]), (200, [25,10])]
# this one draws (approximately) a rectangular path by moving straight for a while, pivoting, and moving straight again
trackSpeedsRect = [(100, [20,20]), (100, [-7,7]), (100, [20, 20]), (100, [-7,7]), (200, [20,20])]
# this one spins for a long time and then drives out in some radial direction. (You might want to try this offline,
# it spins for ~15 minutes!)
trackSpeedsSpin = [(10000, [-10,10]), (200, [10,10])]

# arguments: first (required) argument is the trajectory you want.
# second argument is the logfile name
# third argument is the number of repeats. Set to 1 to do empirical measurement,
# set to some multiple if you want the same motion repeated with automatic pose
# measurement. Cozmo is designed here to return to its start pose between runs so
# results should be *relatively* consistent. Some empirical measurement may also
# be necessary here!

trackSpeeds = trackSpeedsCircle # Had to mdify to allow for default circle option as current structure required the additional argument
if sys.argv[1] == "Circle":
   trackSpeeds = trackSpeedsCircle
elif sys.argv[1] == "Curvy":
   trackSpeeds = trackSpeedsCurvy
elif sys.argv[1] == "Rect":
   trackSpeeds = trackSpeedsRect
elif sys.argv[1] == "Spin":
   trackSpeeds = trackSpeedsSpin
if len(sys.argv) >= 3:
	logName = sys.argv[2]+".py"
else:
	logName = "motionLog.py"
if len(sys.argv) >= 4:
        repeats = int(sys.argv[2])
else:   repeats = 1


async def cozmo_program(robot: cozmo.robot.Robot):
    startPose = robot.pose # return position for multiple runs
    for r in range(repeats):
        for m in trackSpeeds:
            for t in range(m[0]):
                #print("Robot pose: " + str(robot.pose))
                robotPose = Frame2D.fromPose(robot.pose)
                # set the motion for this part of the cycle
                if t == 0:
                    robot.drive_wheel_motors(m[1][0], m[1][1])
                #print("Robot pose 2D frame: " + str(robotPose))
                robotFrames.append((t,robotPose))
                #print()
                await asyncio.sleep(0.1)
        robot.stop_all_motors()
        if (r+1) < repeats:
            returned = robot.go_to_pose(startPose)
            returned.wait_for_completed()

    logFile = open(logName, 'w')

    print("from frame2d import Frame2D", file=logFile)
    print("robotFrames = [", file=logFile)
    for idx in range(len(robotFrames)):
        t = robotFrames[idx][0]
        x = robotFrames[idx][1].x()
        y = robotFrames[idx][1].y()
        a = robotFrames[idx][1].angle()
        print("   (%d, Frame2D.fromXYA(%f,%f,%f))" % (t,x,y,a), end="", file=logFile)
        if idx != len(robotFrames)-1:
            print(",", file=logFile)
    print("]", file=logFile)


cozmo.robot.Robot.drive_off_charger_on_connect = True
cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=False)



