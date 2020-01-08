import numpy as np

# need to invert to end effector cordinates
def rodrigues(v, k, theta):
    term1 = v*np.cos(theta)
    term2 = np.cross(k,v)*np.sin(theta)
    term3 = k*(k @ v)*(1-np.cos(theta))
    v_rot = term1+term2+term3
    return v_rot

def rotation_matrix_from_axis_angle(axis_angle):
    theta = np.linalg.norm(axis_angle) % (2*np.pi)
    unit = axis_angle/theta
    col1 = rodrigues(np.array([1,0,0]), unit, theta)
    col2 = rodrigues(np.array([0,1,0]), unit, theta)
    col3 = rodrigues(np.array([0,0,1]), unit, theta)
    R = np.vstack([col1, col2, col3]).T
    R[np.abs(R)<1e-16] = 0
    return R

def make_homogenous_transformation_matrix(R,p):
    H = np.zeros([4,4])
    H[:3,:3] = R
    H[:3,3] = p
    H[3,3] = 1
    return H

def make_inverse_homogenous_transformation_matrix(H):
    R = H[:3,:3]
    p = H[:3,3]
    H = np.zeros([4,4])
    H[:3,:3] = R.T
    H[:3,3] = -(R.T @ p)
    H[3,3] = 1
    return H

def rotation_about_x(theta):
    R = np.array([[1,             0,              0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def rotation_about_y(theta):
    R = np.array([[np.cos(theta),  0,  np.sin(theta)],
                  [0,              1,              0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def rotation_about_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [            0,              0, 1]])
    return R

def rotation_matrix_to_axis_angle(R):
    """
    R (ndarray): A 3,3 rotation matrix

    Returns: An ndarray with shape (3,) that represents axis angle

    This function takes in a rotation in rotation matrix representation and
    turns it into axis angle representation

    This particular implementation will have issues if the rotation is magnitude
    is exactly equal to 0 or Pi. The 0 case will mean that the vector has norm
    0, but the 180 degree case will cause the same. If in doubt calculate what
    angle you expect and see if a more sophisticated algorihtm is needed.
    """

    theta = np.arccos((R.trace() - 1) /2)
    vec = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    u = vec/np.linalg.norm(vec)

    return u * theta

def make_shifted_axis_angle(rotation_function, theta, H):
    R_shift = rotation_function(theta)
    H_shift = make_homogenous_transformation_matrix(R_shift, np.zeros(3))
    new_H = H @ H_shift
    new_axis_angle = rotation_matrix_to_axis_angle(new_H[:3,:3])
    return new_axis_angle

def make_perturbed_file(uid, finger_cal, position, rotation, fingers, verbose=True):
    """
    Uid is an int
    all other elements are lists
    measurements are given in millimeters and returned in meters
    """
    position = np.array(position)*1e-3
    rotation = np.array(rotation)

    R = rotation_matrix_from_axis_angle(rotation)
    H = make_homogenous_transformation_matrix(R,position)
    H_inv = make_inverse_homogenous_transformation_matrix(H)

    min_val = 0.005 # meters
    max_val = 0.100 # meters
    exps = np.log(min_val), np.log(max_val)
    exps = np.linspace(exps[0],exps[1],7)
    vals_space = np.e**exps
    if verbose:
        print('Shifts in spatial (m): ', vals_space)

    x_dir = -1 if (H @ np.array([max_val,0,0,1]))[2]<0.06 else 1
    y_dir = -1 if (H @ np.array([0,max_val,0,1]))[2]<0.06 else 1

    x_shifts = [(H @ np.array([x_dir*delta,0,0,1]))[:3] for delta in vals_space]
    y_shifts = [(H @ np.array([0,y_dir*delta,0,1]))[:3] for delta in vals_space]
    z_shifts = [(H @ np.array([0,0,-delta,1]))[:3] for delta in vals_space]
    shifts = [x_shifts, y_shifts, z_shifts]

    min_val = 5 # degrees
    max_val = 45 # degrees
    exps = np.log(min_val), np.log(max_val)
    exps = np.linspace(exps[0],exps[1],7)
    vals_rot = np.e**exps
    if verbose:
        print('Shifts in rotational (degrees): ', vals_rot)

    dirs = []
    for func in (rotation_about_x, rotation_about_y, rotation_about_z):
        tmp = make_shifted_axis_angle(func, max_val, H)
        theta = np.linalg.norm(tmp)%(2*np.pi)
        if (rodrigues(np.array([0,0,-0.25]), tmp/theta, theta)[2]+H[2,3])<0.06:
            direction = -1
        else:
            direction = 1
        dirs.append(direction)
    rx_dir, ry_dir, rz_dir = dirs

    rx_shifts = [make_shifted_axis_angle(rotation_about_x, rx_dir*theta, H) for theta in np.deg2rad(vals_rot)]
    ry_shifts = [make_shifted_axis_angle(rotation_about_y, ry_dir*theta, H) for theta in np.deg2rad(vals_rot)]
    rz_shifts = [make_shifted_axis_angle(rotation_about_z, rz_dir*theta, H) for theta in np.deg2rad(vals_rot)]

    with open('{}-ur5.csv'.format(uid),'w') as f:
        position = list(position)
        rotation = list(rotation)
        #f.write('Fingers at calibration = {}\n'.format(finger_cal))
        f.write('No Perturbation,{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format( *(position+rotation+fingers+finger_cal) ))

        for shift, val, axis in zip(x_shifts[::-1]+y_shifts[::-1]+z_shifts[::-1], (list(vals_space)[::-1])*3, ['x']*7+['y']*7+['z']*7):
            shift = list(shift)
            f.write('{} {},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format( axis, round(val,4), *(shift+rotation+fingers+finger_cal) ))

        for shift, val, axis in zip(rx_shifts[::-1]+ry_shifts[::-1]+rz_shifts[::-1], (list(vals_rot)[::-1])*3, ['rx']*7+['ry']*7+['rz']*7):
            shift = list(shift)
            f.write('{} {},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format( axis, round(val,4), *(position+shift+fingers+finger_cal) ))

    #print(H @ np.array([0, 0, -0.005, 1]))
    #print(H_inv @ np.concatenate([position,[1]]))
if __name__ == '__main__':
    # uid = 105193
    # finger_cal = [14533, 14951, 16568, 16664]
    # position = [3.438576,-58.983569,116.676225]
    # rotation = [-1.236684,1.169486,1.107952]
    # fingers = [16585,12901,18609,16276]
    # make_perturbed_file(uid, finger_cal, position, rotation, fingers)

    grasps_to_perturb = np.loadtxt('Rombokas/sugarbox_grasps.csv', delimiter=',', skiprows=1, usecols=(1,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    for i in range(grasps_to_perturb.shape[0]):
        uid = int(grasps_to_perturb[i][0])
        finger_cal = list(grasps_to_perturb[i][1:5])
        position = list(grasps_to_perturb[i][5:8])
        rotation = list(grasps_to_perturb[i][8:11])
        fingers = list(grasps_to_perturb[i][11:])
        make_perturbed_file(uid, finger_cal, position, rotation, fingers, verbose=False)
