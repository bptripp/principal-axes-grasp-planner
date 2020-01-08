import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import point_cloud.mesh_classes as mc
from point_cloud.helper import *
from point_cloud.mind_the_gap import find_mirror_point_cloud
from glob import glob
import matplotlib.pyplot as plt
show_plots = False

data = pd.read_csv('gripper_pose (rev 1 - closest center).tsv', sep='\t')

# Build the rotation and translation to the center of the table
R1 = rotate_x(np.deg2rad(-102))
R2 = rotate_y(np.deg2rad(-0.9))
R3 = rotate_z(np.deg2rad(-135))
R = R3 @ R2 @ R1
p = np.array([[-35.3906], [28.6732], [22.75]])*1e-2

R_inv = R.T
p_inv = -R_inv @ p

for folder_fname in glob('Images/*'):
    name = folder_fname.split('/')[0]
    new_data = []
    for img_fname in glob(folder_fname+'/*.npy'):
        id_ = int(img_fname.split('/')[-1].split('_')[0])
        print(img_fname)
        img = mc.DepthImage(np.load(img_fname).astype(np.float32))
        point_cloud = get_object_point_cloud(img, R, p)
        Q, _ = find_mirror_point_cloud(point_cloud, R_inv, p_inv)
        o_points = point_cloud.get_points()
        q_points = Q.get_points()
        points = np.concatenate([o_points, q_points])

        xy = points[:, (0,1)]
        xy_mean = xy.mean(axis=0)
        z_mean = points[:,2].mean()
        X = (xy - xy_mean).T
        C = X @ X.T
        vals, vecs = np.linalg.eig(C)
        ind = np.argmax(vals)
        vec = vecs[:,ind]
        vec = vec/np.linalg.norm(vec)
        if np.dot(vec, np.array([1,0]))>0:
            # Try to guarantee we grab from robot side
            vec = -vec
        x_vec = vec
        y_vec = np.array([-vec[1], vec[0]])
        rotmat = np.array([x_vec, y_vec]).T

        X_rectified = rotmat.T @ X
        bounds = []
        for i in range(2):
            mask = (X_rectified[i,:]<np.percentile(X_rectified[i,:], 95)) & (X_rectified[i,:]>np.percentile(X_rectified[i,:], 5))
            r = np.max(X_rectified[i,mask]) - np.min(X_rectified[i,mask])
            bounds.append(r)
        bounds.append(np.max(points[:,2]) - np.min(points[:,2]))

        if z_mean<0.06:
            # Pick from the top
            p_obj = np.concatenate([xy_mean, [0.125]])
            R_obj = np.array([[x_vec[0], -y_vec[0],  0],
                              [x_vec[1], -y_vec[1],  0],
                              [       0,         0,  -1]])
        else:
            # Pick from the side
            p_obj = np.concatenate([y_vec*(bounds[1]+0.01)+xy_mean, [z_mean]])
            R_obj = np.array([[       0, x_vec[0],  -y_vec[0]],
                              [       0, x_vec[1],  -y_vec[1]],
                              [      -1,        0,         0]])

        rot_vec = Rotation.from_matrix(R_obj).as_rotvec()
        line = data[data.id==id_].values[0]
        line[8:11] = p_obj*1000
        line[11:14] = rot_vec
        new_data.append(line)

        if show_plots:
            verteces = np.array([(bounds[0]/2,bounds[1]/2), (-bounds[0]/2,bounds[1]/2) , (-bounds[0]/2,-bounds[1]/2), (bounds[0]/2,-bounds[1]/2)])
            verteces = (rotmat @ verteces.T).T + xy_mean

            plt.figure(figsize=(9,9))
            plt.scatter(points[:,0], points[:,1])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.plot([xy_mean[0], xy_mean[0]+x_vec[0]*1e-1], [xy_mean[1], xy_mean[1]+x_vec[1]*1e-1], 'r')
            plt.plot([xy_mean[0], xy_mean[0]+y_vec[0]*1e-1], [xy_mean[1], xy_mean[1]+y_vec[1]*1e-1], 'g')
            plt.scatter(verteces[:,0], verteces[:,1])
            plt.scatter(p_obj[0], p_obj[1], marker='*', color='purple')
            plt.xlim([-0.1, 0.1])
            plt.ylim([-0.1, 0.1])
            plt.gca().set_aspect('equal')

            plt.show()
    new_data = pd.DataFrame(new_data, columns=data.columns)
    new_data.to_csv(name+'_grasps.csv')
