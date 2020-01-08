import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from . import mesh_classes as mc

DEPTH2METERS = 0.00012498664727900177

def depth2meters(dimg):
    """
    Takes an ndarray of raw depth values from the RealSense SR300 and returns
    a new array with depth values in meters

    Inputs
    ------
    dimg (ndarray)

    Outputs
    -------
    res (ndarray): same dimensions as input
    """
    return 0.00012498664727900177*dimg

def get_points_of_center_object(point_cloud, eps_dbscan=0.01,
                                min_samples_dbscan=20, point_threshold=200):
    """
    Given a PointCloud object we use DBSCAN to find the combination of points
    that make up the object closest to the origin. We apply DBSCAN only on the
    x-y points (ie the top view), as we found that this tends to give better
    results. Be warned that the origin will obviously depend on which coordinate
    system you are representing your point cloud in.
    It returns a PointCloud object of the centermost object.

    eps_dbscan is the max distance to search for neighbouring points in DBSCAN.
    The default value is set to 0.01 expecting to the point cloud to be in
    meters.

    min_samples_dbscan is the minimum number of points needed to consider a
    point a core point under DBSCAN. The default is set to 20.

    point_threshold is the minimum number of points we require for an object. If
    it has less than 200 points (the default) we ignore it (probably noise).
    """
    points = point_cloud.get_points()
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan).fit(points[:,(0,1)])
    labels_d = dbscan.labels_

    min_points = None
    min_dist = np.inf
    for label in set(labels_d):
        tmp = points[labels_d==label]
        if label==-1 or tmp.shape[0]<point_threshold:
            continue
        dist = np.linalg.norm(tmp.mean(axis=0)[:2])
        if dist<min_dist:
            min_points = tmp
            min_dist = dist

    return mc.PointCloud(min_points)

def get_object_point_cloud(dimg, R, trans=np.array([4.75, -45.3, 22.75])*1e-2,
                           inner_distance=0.3, outer_distance=0.8,
                           eps_dbscan=0.01, min_samples_dbscan=20):
    """
    Given a DepthImage, find the centermost object after applying a series of
    preprocessing and coordinate transformation steps.

    First we remove all points outside of the range inner_distance to outer_distance,
    and project the depth image into a point cloud. Then we apply the rotation given.
    We finish the coordinate transformation by translating the point cloud by trans.
    Finally we apply DBSCAN and pick the cluster that is closest to the origin.

    All the default values set here are ones that have been found to work with
    our data collection set-up.

    Returns a point_cloud with the center object in table coordinates
    """

    # Only take points between inner_distance meters and
    # outer_distance meters away from the camera and smooth
    # with a median filter
    med_dimg = dimg.depth_threshold(inner_distance, outer_distance, filter=True)

    # Get the point cloud
    pts=med_dimg.project_points()

    # Apply coordinate transform to point cloud.
    pts.coordinate_transform(R, trans)

    # Get only the points on the table and above it
    box_pts = pts.box_threshold( xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), zlim=(0, 0.5))

    center_object = get_points_of_center_object(box_pts)
    return center_object

def rotate_x(theta):
    R = np.array([[1, 0, 0],
                  [0,np.cos(theta),-np.sin(theta)],
                  [0,np.sin(theta), np.cos(theta)]])
    return R

def rotate_y(theta):
    R = np.array([[np.cos(theta),0,np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta),0, np.cos(theta)]])
    return R

def rotate_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [            0,              0, 1]])
    return R

def generate_potential_grasp(object_cloud):
    """
    The object_cloud needs to be in table coordinates.
    """
    # https://www.cs.princeton.edu/~funk/tog02.pdf picking points in triangle
    nrmls = object_cloud.normals.copy()
    # if object_cloud.points[:,2].max()<0.11:
    #     nrmls[nrmls[:,2]>0] *= -1
    #     direction_bias = np.max( np.vstack( [ nrmls @ np.array([0,0,-1]), np.zeros(nrmls.shape[0])] ), axis=0 )
    # else:
    #     direction_bias = np.ones(nrmls.shape)
    area_bias = object_cloud.facet_areas/np.sum(object_cloud.facet_areas)
    probability = area_bias
    probability /= np.sum(probability)

    sample = np.random.choice(np.arange(object_cloud.hull.simplices.shape[0]), p=probability)
    simplex = object_cloud.hull.simplices[sample]
    r1,r2 = np.random.uniform(0,1,2)
    sqrt_r1 = r1**0.5
    A,B,C = object_cloud.points[simplex]
    point = (1-sqrt_r1)*A + sqrt_r1*(1-r2)*B + sqrt_r1*r2*C

    direction = nrmls[sample] # this is pointing inwards
    distance = np.random.uniform(0.01, 0.15) # in cm

    p = point - direction*distance
    if p[2] < 0.07:
        n = (point[2] - 0.07)/distance - direction[2]
        direction[2] = direction[2]+n
        direction = direction/np.linalg.norm(direction)
        p = point - direction*distance

    y_axis = np.random.uniform(-1,1,3)
    y_axis = y_axis - (y_axis@direction)*direction
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, direction)
    x_axis /= np.linalg.norm(x_axis)

    R = np.zeros((3,3))
    R[:,0] = x_axis
    R[:,1] = y_axis
    R[:,2] = direction
    return R, p[...,np.newaxis]

def plot_object_and_gripper(object_cloud, gripper, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for simplex in object_cloud.hull.simplices:
        poly = object_cloud.points[simplex]
        x,y,z = poly[:,0], poly[:,1], poly[:,2]
        ax.plot_trisurf(x, y, z)

    gripper.make_new_shell(apply_pose=True)
    for simplex in gripper.shell.hull.simplices:
        poly = gripper.shell.points[simplex]
        x,y,z = poly[:,0], poly[:,1], poly[:,2]
        ax.plot_trisurf(x, y, z, alpha=0.5)

    dummy = np.ones([120,4])
    thetas = gripper._final_thetas

    transforms = [gripper.H_finger_1, gripper.H_finger_2, gripper.H_finger_3]
    for i,theta in enumerate(thetas):
        flexed = gripper.finger_flex(theta)
        dummy[:,:3] = flexed
        finger = (gripper.pose @ (transforms[i] @ dummy.T)).T
        ax.scatter(finger[:,0], finger[:,1], finger[:,2])

    p = gripper.position
    R = gripper.orientation
    ax.scatter(*p)
    x = p[:,0]+0.05*R[:,0]
    y = p[:,0]+0.05*R[:,1]
    z = p[:,0]+0.05*R[:,2]
    ax.plot( *zip(p[:,0],x), color='black')
    ax.plot( *zip(p[:,0],y), color='green')
    ax.plot( *zip(p[:,0],z), color='blue')
    ax.set_xlim([-0.2,0.2])
    ax.set_ylim([-0.2,0.2])
    ax.set_zlim([0,0.2])

    if show:
        plt.show()
