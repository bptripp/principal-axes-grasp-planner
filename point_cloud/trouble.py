import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mesh_classes as mc
from helper import *
from mind_the_gap import find_mirror_point_cloud
from gripper import ReFlex

img = mc.DepthImage(np.load('Images27June/27Jun2018/481173_RS_depth.npy').astype(np.float32))

# Build the rotation and translation to the center of the table
R1 = rotate_x(np.deg2rad(-102))
R2 = rotate_y(np.deg2rad(-0.9))
R3 = rotate_z(np.deg2rad(-135))
R = R3 @ R2 @ R1
p = np.array([[-35.3906], [28.6732], [22.75]])*1e-2

point_cloud = get_object_point_cloud(img, R, p)

R_inv = R.T
p_inv = -R_inv @ p
Q, _ = find_mirror_point_cloud(point_cloud, R_inv, p_inv)

o_points = point_cloud.get_points()
q_points = Q.get_points()

# Plot Cross sections
fig = plt.figure()
plt.subplot(221)
plt.title('Top View')
plt.scatter(o_points[:,0], o_points[:,1])
plt.scatter(q_points[:,0], q_points[:,1])

plt.subplot(222)
plt.title('Side View')
plt.scatter(o_points[:,1], o_points[:,2])
plt.scatter(q_points[:,1], q_points[:,2])

plt.subplot(223)
plt.title('Front View')
plt.scatter(o_points[:,0], o_points[:,2])
plt.scatter(q_points[:,0], q_points[:,2])

object_cloud = mc.PointCloud(np.concatenate([o_points, q_points]), form_hull=True)

#Plot hull
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
for simplex in object_cloud.hull.simplices:
    poly = object_cloud.points[simplex]
    x,y,z = poly[:,0], poly[:,1], poly[:,2]
    ax.plot_trisurf(x, y, z)
plt.show()
