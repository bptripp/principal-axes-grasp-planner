import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import find_objects
from . import mesh_classes as mc
from . import helper as h

def find_mirror_point_cloud(point_cloud, R, p, n=20, m=50):
    """
    point_cloud must be in table coordinates

    R and p are the rotation and translation to get us from table to camera
    coordinates.

    Returns Q in table coordinates
    """
    Qs = generate_hypotheses(point_cloud.points, n=n, m=m)
    points_dimg = point_cloud.back_project(R, p)

    scores = list(map(lambda Q: score_hypothesis(points_dimg, Q, R, p), Qs))
    ind = np.argmin(scores)
    return Qs[ind], scores[ind]

def generate_hypotheses(points, trans=np.array([[4.75], [-45.3], [22.75]])*1e-2, n=20, m=50):
    mu = points.mean(axis=0)
    X = points - mu
    extent = abs(max( np.max(X[:,0])-np.min(X[:,0]), np.max(X[:,1])-np.min(X[:,1])))

    _, eig_vectors = np.linalg.eig(X[:,:2].T @ X[:,:2])

    thetas = np.linspace(-20,20, n//2)
    ds = np.linspace(-0.5*extent, 0.5*extent, m//2)

    hypotheses = []
    for i in range(2):
        normal = np.concatenate([eig_vectors[:,i],[0]])
        for theta in thetas:
            r = h.rotate_z(np.deg2rad(theta))
            for d in ds:
                Q = np.array((-1, -1,1))*(r @ (X + d*normal).T).T + mu
                hypotheses.append(mc.PointCloud(Q))

    return hypotheses

def score_hypothesis(object_dimg, Q, R, trans):
    """
    Points need to be in camera coordinates
    All distances are in meters
    """
    if Q.points.shape[0]>3000:
        indeces = np.random.randint(0,Q.points.shape[0],[3000])
        Q = mc.PointCloud(Q.points[indeces])
    Q_img = Q.back_project(R, trans).dimg # the raw ndarray is used here
    object_dimg = object_dimg.dimg

    # Points outside mask
    if np.sum((Q_img>0) & (object_dimg==0))>0:
        comb = (object_dimg>0) | (Q_img>0)
        slice_x, slice_y = find_objects(comb)[0]
        dist = edt(object_dimg[slice_x,slice_y]==0)
        q_slice = Q_img[slice_x, slice_y]
        o_slice = object_dimg[slice_x,slice_y]
        first_dim, second_dim = np.nonzero((q_slice>0) & (o_slice==0))
        score1 = np.mean(dist[first_dim, second_dim])
    else:
        score1 = 0

    # Points in front of mask
    nonzero_closer = (Q_img!=0) & (object_dimg>Q_img)
    score2 = object_dimg[nonzero_closer] - Q_img[nonzero_closer]
    if score2.shape[0]>0:
        score2 = np.mean(score2)
    else:
        score2 = 0

    return score1 + 2000*score2
