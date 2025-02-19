# Localization and Navigation on the Anki Cozmo Robot

## Overview
This project was part of the Autonomous Intelligent Systems (COMP7040) module at Oxford Brookes University. The focus was on developing a robust localization and navigation system for the Anki Cozmo robot using probabilistic sensor models, Monte-Carlo Localization (MCL), and motion planning techniques.

## Objectives
The project aimed to:
- Develop an understanding of autonomous self-navigating robots, including planning, localization, and control.
- Build accurate sensor and motion models for the Cozmo robot.
- Utilize a simulation environment to refine navigation algorithms.
- Implement Monte-Carlo Localization (MCL) to estimate the robot's position.
- Navigate to a goal while dealing with uncertainty in sensors and actuators.

This work contributed to areas such as autonomous cars, drones, and robotic delivery systems, offering valuable insights into real-world AI challenges.

## Learning Outcomes
Through this project, we gained experience in:
1. Evaluating system intelligence requirements in autonomous systems.
2. Collaborating on software and hardware development for intelligent robotics.
3. Developing independence and professional awareness, including ethical and risk considerations.
4. Producing technical documentation to support autonomous system development.

## Robot Description
The Anki Cozmo is a tracked mobile robot equipped with:
- A camera for recognizing landmarks and faces.
- A cliff sensor to detect edges.
- A lifter for manipulating objects (Light Cubes).
- A Python SDK for programming and simulation.

A provided simulation environment replicated Cozmo’s kinematics and basic physics, enabling controlled experiments.

## Project Components
### 1. Motion Model
- Implemented the robot's differential drive kinematics.
- Integrated the motion model into `track_speed_to_pose_change` in `cozmo_interface.py`.
- Experimentally determined the optimal wheel distance parameter.

### 2. Probabilistic Sensor Model
- Developed a Gaussian sensor model by estimating standard deviations for x/y/theta.
- Implemented a sensor model for visual cube detection in `cube_sensor_model`.

### 3. Monte-Carlo Localization
- Implemented the MCL algorithm using `run-mcl.py` and `mcl_tools.py`.
- Optimized particle count and other parameters to improve localization accuracy.

### 4. Robot-Go-Home Challenge
- Developed a navigation system that drove the robot to a goal from any initial position while avoiding obstacles.
- Experimentally assessed the effectiveness of the solution.
- Discussed legal, ethical, and societal implications of autonomous robotics.

## Setup and Requirements
### Prerequisites
- Python 3.7+
- Cozmo SDK
- NumPy, OpenCV, Matplotlib (for visualization and data processing)
- A working installation of the Cozmo simulator

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/cozmo-localization.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the simulation:
   ```bash
   python run-mcl.py
   ```

## License
Copyright (c) 2019 Matthias Rolf, Oxford Brookes University
This project was for educational purposes. All rights reserved by Oxford Brookes University.

---
For more details, refer to the [Oxford Brookes Academic Misconduct Policy](https://www.brookes.ac.uk/students/sirt/student-conduct/academic-misconduct/).

