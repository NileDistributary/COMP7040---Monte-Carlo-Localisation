import numpy as np
from frame2d import Frame2D
from motionLog1 import robotFrames
import matplotlib.pyplot as plt

# We are using trackSpeedsCircle
track_speeds = np.array([[10, 25]] * 600)  # 600 steps of [10, 25]

def test_wheel_distance(wheel_distances, track_speeds, actual_frames, time_interval=0.1):
    results = {}
    
    # Get initial and final actual poses
    initial_pose = actual_frames[0][1]
    final_actual = actual_frames[-1][1]
    for wheel_dist in wheel_distances:
        # Define modified track_speed_to_pose_change with specified wheel distance
        def ts_to_pose_change(left, right, time):
            if left == right:
                x = left * time
                y = 0
                a = 0
            else:
                theta = (right - left) / wheel_dist
                r = (left + right) / (2 * theta)
                x = r * np.sin(theta)
                y = -r * (np.cos(theta) - 1)
                a = theta
            return Frame2D.fromXYA(x, y, a)
        
        # Simulate robot movement with this wheel distance
        predicted_pose = initial_pose
        
        for speeds in track_speeds:
            left, right = speeds
            pose_change = ts_to_pose_change(left, right, time_interval)
            predicted_pose = predicted_pose.mult(pose_change)
        
        # Calculate error between predicted and actual final position
        position_error = np.sqrt((predicted_pose.x() - final_actual.x())**2 + 
                                (predicted_pose.y() - final_actual.y())**2)
        
        angle_diff = predicted_pose.angle() - final_actual.angle()
        angle_error = abs((angle_diff + np.pi) % (2 * np.pi) - np.pi)
        
        results[wheel_dist] = {
            'position_error': position_error,
            'angle_error': angle_error,
            'combined_error': position_error + angle_error * 50  # Weighting factor
        }
    
    return results

# Test with wheel distance values from 40 to 100 mm in 5 mm increments
wheel_distances = range(40, 100, 5)
results = test_wheel_distance(wheel_distances, track_speeds, robotFrames)

# Plot the results
distances = list(results.keys())
position_errors = [results[d]['position_error'] for d in distances]
angle_errors = [results[d]['angle_error'] for d in distances]

plt.figure(figsize=(10, 6))
plt.plot(distances, position_errors, 'bo-', label='Position Error (mm)')
plt.plot(distances, [a * 100 for a in angle_errors], 'ro-', label='Angle Error (rad×100)')
plt.xlabel('Wheel Distance (mm)')
plt.ylabel('Error')
plt.title('Error vs Wheel Distance Parameter')
plt.legend()
plt.grid(True)
plt.savefig('plots/Wheel Distance Error/wheeldistanceerrorformotionLog1.png')
plt.show()
