import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mesh_classes as mc
from helper import *
from gripper import ReFlex

# Make a cube
points = np.array([[-50.001, -50.001,  0],
                   [-50.001,  50.001,  0],
                   [ 50.001,  50.001,  0],
                   [ 50.001, -50.001,  0],
                   [-50.002, -50.002, 80],
                   [-50.002,  50.002, 80],
                   [ 50.002,  50.002, 80],
                   [ 50.002, -50.002, 80]])*1e-3

object_cloud = mc.PointCloud(points, form_hull=True)

# Search for a pose
gripper = ReFlex()
searching = True
while searching:
    R, p = generate_potential_grasp(object_cloud)
    gripper.set_gripper_pose(R, p)
    searching = not(gripper.check_valid_grasp(object_cloud))

# Plot Pose
plot_object_and_gripper(object_cloud, gripper)
